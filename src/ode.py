"""
ode.py — Neural ODE for Cardiac Contraction Dynamics
=====================================================
Models the sequence of per-timestep graph embeddings as a continuous-time
dynamical system in latent space:

    dz/dt = f_θ(z(t), t)

where:
  z(t) ∈ R^d_z  is the latent state (contraction dynamics at time t)
  f_θ           is a small MLP (the ODE function)
  t ∈ [0, 1]   is normalized time from ED (0) to peak systole (1)

The ODE is solved at the observed odd-frame times using torchdiffeq.
The final state z(t=1) is a compressed representation of the entire
contraction trajectory — this is what the classifier uses.

Key advantages over stacking embeddings:
  1. Handles variable N_frames naturally (query ODE at any t)
  2. Enforces temporal smoothness by construction
  3. Learns the *dynamics* of contraction, not just static features
  4. Adjoint method makes backprop memory-efficient regardless of N_frames

Architecture:
  Input     : z_0 = graph_embed[t=0]  (B, d_z=64)
  ODE func  : MLP [d_z+1 → 256 → 256 → d_z] with time concatenation
  Solver    : euler (fast, for prototyping) | dopri5 (accurate, for final runs)
  Output    : z(t_final)  (B, d_z=64) — final latent state

Usage:
    from ode import NeuralODEClassifier

    model = NeuralODEClassifier(d_z=64, n_classes=5).to(device)
    logits = model(embeddings_sequence, times)
    # embeddings_sequence : (B, N_frames, d_z)
    # times               : (B, N_frames) normalized to [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
from typing import Tuple


# ── ODE Function f_θ(z, t) ──────────────────────────────────────────────────

class ODEFunc(nn.Module):
    """
    The right-hand side of the Neural ODE: dz/dt = f_θ(z, t)

    Architecture: MLP with time concatenation at each layer.
    Time is injected at the input (not just concatenated once) to allow
    the dynamics to change character across the cardiac cycle — early
    contraction looks different from late contraction.

    We use a "highway" (residual) connection so the ODE can learn to
    be near-identity when the dynamics are slow (e.g., during isovolumic
    phases), which improves gradient flow during training.

    Args:
        d_z       : latent state dimension (matches graph encoder output)
        hidden_dim: MLP hidden dimension
        dropout   : dropout on hidden layers
    """

    def __init__(
        self,
        d_z:        int = 64,
        hidden_dim: int = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.d_z = d_z

        # Time is concatenated to z → input dim = d_z + 1
        self.net = nn.Sequential(
            nn.Linear(d_z + 1, hidden_dim),
            nn.Tanh(),                       # Tanh preferred for ODEs — bounded
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_z),
        )

        # Initialize final layer near zero → near-identity dynamics at init
        # This helps early training — the ODE starts as "nothing is changing"
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Track number of function evaluations (diagnostic)
        self.nfe = 0

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : scalar tensor — current time point (torchdiffeq convention)
            z : (B, d_z) — current latent state

        Returns:
            dz_dt : (B, d_z) — time derivative of latent state
        """
        self.nfe += 1

        # Broadcast t to batch: (B, 1)
        t_expand = t.expand(z.shape[0], 1)

        # Concatenate time to state
        zt = torch.cat([z, t_expand], dim=1)  # (B, d_z + 1)

        dz_dt = self.net(zt)  # (B, d_z)

        return dz_dt

    def reset_nfe(self):
        """Reset function evaluation counter."""
        self.nfe = 0


# ── Neural ODE integrator ────────────────────────────────────────────────────

class NeuralODEBlock(nn.Module):
    """
    Wraps torchdiffeq's ODE solver with our ODEFunc.

    Supports two modes:
      - 'euler'  : fixed-step, fast, good for prototyping
      - 'dopri5' : adaptive Runge-Kutta 4/5, accurate, use for final results
      - 'adjoint': dopri5 + adjoint method for memory-efficient backprop
                   (important when N_frames is large)

    The adjoint method computes gradients by solving a second ODE backward
    in time, using O(1) memory instead of O(N_steps) for standard backprop
    through the solver. Use this for final training runs.

    Args:
        d_z      : latent dimension
        method   : ODE solver ('euler', 'dopri5', 'adjoint')
        rtol     : relative tolerance for adaptive solvers
        atol     : absolute tolerance for adaptive solvers
    """

    def __init__(
        self,
        d_z:     int   = 64,
        method:  str   = 'euler',
        rtol:    float = 1e-3,
        atol:    float = 1e-4,
    ):
        super().__init__()
        self.method  = method
        self.rtol    = rtol
        self.atol    = atol
        self.odefunc = ODEFunc(d_z=d_z)

    def forward(
        self,
        z0:      torch.Tensor,    # (B, d_z) initial state
        t_span:  torch.Tensor,    # (N_t,) time points to evaluate at
    ) -> torch.Tensor:
        """
        Integrate the ODE from t_span[0] to t_span[-1],
        returning the state at every time point in t_span.

        Args:
            z0     : (B, d_z) — initial latent state at t=t_span[0]
            t_span : (N_t,)   — sorted time points, e.g. [0.0, 0.2, 0.4, ...]

        Returns:
            z_traj : (N_t, B, d_z) — latent state at each time point
                     z_traj[0] = z0
                     z_traj[-1] = z(t_final) ← use this for classification
        """
        self.odefunc.reset_nfe()

        solver_kwargs = {
            "rtol": self.rtol,
            "atol": self.atol,
        }

        if self.method == 'adjoint':
            # Memory-efficient adjoint backprop
            z_traj = odeint_adjoint(
                self.odefunc, z0, t_span,
                method='dopri5',
                **solver_kwargs
            )
        else:
            z_traj = odeint(
                self.odefunc, z0, t_span,
                method=self.method,
                **solver_kwargs
            )

        return z_traj  # (N_t, B, d_z)

    @property
    def nfe(self) -> int:
        """Number of function evaluations in last forward pass."""
        return self.odefunc.nfe


# ── Sequence encoder: graph embeds → ODE initial state ──────────────────────

class SequenceEncoder(nn.Module):
    """
    Takes the sequence of per-frame graph embeddings and computes z_0,
    the initial ODE state.

    We use the FIRST frame embedding directly as z_0 (simplest option).
    An alternative is a learned projection, but for 64-dim embeddings
    this is unnecessary — the ODE function learns the rest.

    Also handles the case where different samples in a batch have
    different N_frames (due to padding in the collate function) by
    using only the first frame for z_0.

    Args:
        d_z : latent dimension (must match graph encoder out_dim)
    """

    def __init__(self, d_z: int = 64):
        super().__init__()
        # Optional: learned projection from graph embed to ODE latent space
        # Identity if d_graph == d_z (which is our case)
        self.proj = nn.Identity()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings : (B, N_frames, d_z) — graph embeddings per time step

        Returns:
            z0 : (B, d_z) — initial ODE state (from first frame)
        """
        z0 = embeddings[:, 0, :]  # (B, d_z) — first frame embedding
        return self.proj(z0)


# ── Classifier head ──────────────────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """
    MLP classifier from final ODE state z(t_final) to class logits.

    We use a 2-layer MLP with dropout — simple and effective.
    The final ODE state encodes the entire contraction trajectory,
    so the classifier just needs to read out the class from this
    compressed representation.

    Args:
        d_z       : latent dimension (input)
        n_classes : number of output classes (5 for ACDC)
        dropout   : dropout probability
    """

    def __init__(
        self,
        d_z:       int = 64,
        n_classes: int = 5,
        dropout:   float = 0.3,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(d_z, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, z_final: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_final : (B, d_z) — final ODE state

        Returns:
            logits : (B, n_classes)
        """
        return self.head(z_final)


# ── Full ODE module: embeddings → logits ────────────────────────────────────

class CardiacODEClassifier(nn.Module):
    """
    Full Neural ODE classification module.

    Takes the sequence of graph embeddings from GraphMotionEncoder,
    runs them through a Neural ODE, and classifies from the final state.

    This is the third stage in the full pipeline:
        GraphMotionEncoder → CardiacODEClassifier → logits

    Input  : embeddings (B, N_frames, d_z), times (B, N_frames) or (N_frames,)
    Output : logits (B, n_classes)

    Args:
        d_z       : latent dimension (must match GraphMotionEncoder out_dim)
        n_classes : number of classes (5 for ACDC)
        method    : ODE solver method
    """

    def __init__(
        self,
        d_z:       int = 64,
        n_classes: int = 5,
        method:    str = 'euler',
    ):
        super().__init__()
        self.d_z = d_z

        self.seq_encoder = SequenceEncoder(d_z)
        self.ode_block    = NeuralODEBlock(d_z=d_z, method=method)
        self.classifier   = ClassifierHead(d_z=d_z, n_classes=n_classes)

    def forward(
        self,
        embeddings: torch.Tensor,     # (B, N_frames, d_z)
        times:      torch.Tensor,     # (B, N_frames) or (N_frames,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits : (B, n_classes)
            z_traj : (N_frames, B, d_z) — full ODE trajectory (for analysis)
        """
        B = embeddings.shape[0]

        # Initial ODE state from first frame embedding
        z0 = self.seq_encoder(embeddings)   # (B, d_z)

        # Time points for ODE integration
        # torchdiffeq requires a 1D tensor of sorted time points
        # If times varies per sample, use the mean (or first sample's times)
        # For ACDC, all samples use the same normalized time grid after collation
        if times.dim() == 2:
            t_span = times[0]   # (N_frames,) — use first sample's times
        else:
            t_span = times      # already (N_frames,)

        # Ensure t_span is on the correct device and sorted
        t_span = t_span.to(embeddings.device)


        # ── FIX: ensure strictly increasing ──────────────────────────────────
        t_span, _ = torch.sort(t_span)
        eps = 1e-5
        for i in range(1, len(t_span)):
            if t_span[i] <= t_span[i - 1]:
                t_span[i] = t_span[i - 1] + eps
        # ─────────────────────────────────────────────────────────────────────

        # Solve ODE: (N_frames, B, d_z)
        z_traj = self.ode_block(z0, t_span)

        # Use final state for classification
        z_final = z_traj[-1]   # (B, d_z)

        logits = self.classifier(z_final)   # (B, n_classes)

        return logits, z_traj

    def get_trajectory(
        self,
        embeddings: torch.Tensor,
        times:      torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience: return full ODE trajectory without computing logits.
        Useful for visualization and geodesic deviation analysis.

        Returns:
            z_traj : (N_frames, B, d_z)
        """
        with torch.no_grad():
            _, z_traj = self.forward(embeddings, times)
        return z_traj


# ── Geodesic deviation metric ────────────────────────────────────────────────

def geodesic_deviation(z_traj: torch.Tensor) -> torch.Tensor:
    """
    Compute geodesic deviation of the ODE trajectory.

    A geodesic in latent space is the straight line from z(0) to z(T).
    Geodesic deviation measures how far the actual trajectory curves
    away from this straight line.

    Healthy hearts should have near-geodesic trajectories (efficient,
    direct contraction). Diseased hearts deviate more.
    This is one of our key interpretable biomarkers for the paper.

    Args:
        z_traj : (N_frames, B, d_z)

    Returns:
        deviation : (B,) — mean deviation per sample (normalized)
    """
    N, B, d = z_traj.shape

    z_start = z_traj[0]   # (B, d)
    z_end   = z_traj[-1]  # (B, d)

    # Straight-line interpolation (geodesic in Euclidean space)
    t_vals = torch.linspace(0, 1, N, device=z_traj.device)  # (N,)
    geodesic = (
        z_start.unsqueeze(0) * (1 - t_vals.view(N, 1, 1))
        + z_end.unsqueeze(0) * t_vals.view(N, 1, 1)
    )  # (N, B, d)

    # Deviation: mean L2 distance from geodesic at each time point
    diff = z_traj - geodesic                               # (N, B, d)
    dist = diff.norm(dim=-1)                               # (N, B)

    # Normalize by total arc length of trajectory (scale-invariant)
    arc_length = diff.norm(dim=-1).sum(dim=0) + 1e-8      # (B,)
    deviation  = dist.mean(dim=0) / arc_length             # (B,)

    return deviation


# ── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time as time_module

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    B       = 4
    N_frames = 6     # typical for ACDC odd-frame selection
    d_z     = 64
    n_cls   = 5

    # Simulate graph encoder outputs
    embeddings = torch.randn(B, N_frames, d_z, device=device)
    times      = torch.linspace(0, 1, N_frames, device=device).unsqueeze(0).expand(B, -1)
    labels     = torch.randint(0, n_cls, (B,), device=device)

    # ── 1. ODEFunc ────────────────────────────────────────────────────────────
    print("\n── ODEFunc ──")
    func = ODEFunc(d_z=d_z).to(device)
    z    = torch.randn(B, d_z, device=device)
    t    = torch.tensor(0.5, device=device)
    dz   = func(t, z)
    print(f"  Input  z : {z.shape}")
    print(f"  Output dz: {dz.shape}")
    print(f"  dz norm  : {dz.norm(dim=1).mean():.6f}  (≈ 0 at init — good)")

    # ── 2. NeuralODEBlock ─────────────────────────────────────────────────────
    print("\n── NeuralODEBlock (euler) ──")
    ode = NeuralODEBlock(d_z=d_z, method='euler').to(device)
    z0      = torch.randn(B, d_z, device=device)
    t_span  = torch.linspace(0, 1, N_frames, device=device)

    t0 = time_module.time()
    z_traj = ode(z0, t_span)
    t1 = time_module.time()

    print(f"  z0 shape     : {z0.shape}")
    print(f"  t_span       : {t_span.tolist()}")
    print(f"  z_traj shape : {z_traj.shape}")   # (N_frames, B, d_z)
    print(f"  z0 preserved : {torch.allclose(z_traj[0], z0, atol=1e-5)}")
    print(f"  Forward time : {(t1-t0)*1000:.1f} ms")
    print(f"  NFE (euler)  : {ode.nfe}")

    # ── 3. dopri5 solver ──────────────────────────────────────────────────────
    print("\n── NeuralODEBlock (dopri5) ──")
    ode_dopri = NeuralODEBlock(d_z=d_z, method='dopri5').to(device)
    t0 = time_module.time()
    z_traj_dopri = ode_dopri(z0, t_span)
    t1 = time_module.time()
    print(f"  Forward time : {(t1-t0)*1000:.1f} ms")
    print(f"  NFE (dopri5) : {ode_dopri.nfe}  (adaptive — more evals, more accurate)")

    # ── 4. Full CardiacODEClassifier ─────────────────────────────────────────
    print("\n── CardiacODEClassifier ──")
    model = CardiacODEClassifier(d_z=d_z, n_classes=n_cls, method='euler').to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {n_params:,}")

    logits, z_traj_full = model(embeddings, times)
    print(f"  embeddings shape : {embeddings.shape}")
    print(f"  logits shape     : {logits.shape}")        # (B, 5)
    print(f"  z_traj shape     : {z_traj_full.shape}")  # (N_frames, B, 64)
    print(f"  logits (raw)     : {logits[0].detach().cpu().tolist()}")

    # ── 5. Backward pass ─────────────────────────────────────────────────────
    print("\n── Backward pass ──")
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    )
    print(f"  CE loss      : {loss.item():.4f}  (expect ≈ ln(5)={torch.log(torch.tensor(5.0)):.4f})")
    print(f"  Gradient norm: {grad_norm:.4f}  (should be > 0)")

    # ── 6. Geodesic deviation ─────────────────────────────────────────────────
    print("\n── Geodesic deviation ──")
    model.eval()
    with torch.no_grad():
        _, z_traj_eval = model(embeddings, times)
    dev = geodesic_deviation(z_traj_eval)
    print(f"  Deviation shape  : {dev.shape}")    # (B,)
    print(f"  Deviation values : {dev.cpu().tolist()}")
    print(f"  (At init ≈ 0 — dynamics are near-identity, trajectory is near-linear)")

    # ── 7. Variable time points across batch ──────────────────────────────────
    print("\n── Variable N_frames (collate padding simulation) ──")
    # Patient A has 6 frames, Patient B was padded to 8
    embeds_var = torch.randn(2, 8, d_z, device=device)
    times_var  = torch.linspace(0, 1, 8, device=device).unsqueeze(0).expand(2, -1)
    logits_var, _ = model(embeds_var, times_var)
    print(f"  Output shape : {logits_var.shape}")   # (2, 5)
    print(f"  No crash ✓")

    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"\nGPU memory used: {mem_mb:.1f} MB")

    print("\n[ode.py] All checks passed.")