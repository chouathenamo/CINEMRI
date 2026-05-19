"""
WITH SKIP CONNECTION
ode.py — Neural ODE for Cardiac Contraction Dynamics  (v4 / skip-connection)
=============================================================================
Changes from v3:
  - CardiacODEClassifier now concatenates z0 (pre-ODE) and z_final (post-ODE)
    before the classifier head — a ResNet-style skip connection.
    · z0      = A_noODE : raw motion dynamics from the GRU (d_z)
    · z_final = A_ODE   : smoothed / continuous dynamics from the ODE (d_z)
    · z_in    = [z0 || z_final || clinical]  → size 2*d_z + n_clinical
  - ClassifierHead in_dim changed accordingly: d_z + n_clinical → 2*d_z + n_clinical
  - Everything else (ODEFunc, NeuralODEBlock, SequenceEncoder) is unchanged.

Motivation:
  Ablation experiments showed the no-ODE variant outperforming the full model,
  suggesting the ODE's smoothed trajectory was overwriting useful pre-ODE
  information rather than complementing it.  By preserving z0 as a skip path,
  the classifier can independently weight raw and smoothed dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
from typing import Optional, Tuple


# ── ODE Function f_θ(z, t) ──────────────────────────────────────────────────

class ODEFunc(nn.Module):
    """
    dz/dt = f_θ(z, t)
    v3: dropout 0.1 → 0.3
    """

    def __init__(self, d_z: int = 64, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.d_z = d_z
        self.net = nn.Sequential(
            nn.Linear(d_z + 1, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_z),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.nfe = 0

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        self.nfe   += 1
        t_expand    = t.expand(z.shape[0], 1)
        return self.net(torch.cat([z, t_expand], dim=1))

    def reset_nfe(self):
        self.nfe = 0


# ── Neural ODE block ─────────────────────────────────────────────────────────

class NeuralODEBlock(nn.Module):

    def __init__(self, d_z: int = 64, method: str = 'euler',
                 rtol: float = 1e-3, atol: float = 1e-4):
        super().__init__()
        self.method  = method
        self.rtol    = rtol
        self.atol    = atol
        self.odefunc = ODEFunc(d_z=d_z)

    def forward(self, z0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        self.odefunc.reset_nfe()
        kwargs = {"rtol": self.rtol, "atol": self.atol}
        if self.method == 'adjoint':
            return odeint_adjoint(self.odefunc, z0, t_span, method='dopri5', **kwargs)
        return odeint(self.odefunc, z0, t_span, method=self.method, **kwargs)

    @property
    def nfe(self) -> int:
        return self.odefunc.nfe


# ── Sequence encoder ─────────────────────────────────────────────────────────

class SequenceEncoder(nn.Module):
    """GRU over graph embedding sequence → z_0 for ODE."""

    def __init__(self, d_z: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size=d_z, hidden_size=d_z,
                          num_layers=1, batch_first=True)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(embeddings)
        return h_n.squeeze(0)


# ── Classifier head ──────────────────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """
    MLP from z_in → logits.

    v4: in_dim = 2*d_z + n_clinical  (was d_z + n_clinical)
        The extra d_z carries the pre-ODE skip connection (z0 = A_noODE).
    Dropout 0.5 unchanged.
    """

    def __init__(self, in_dim: int = 64, n_classes: int = 5, dropout: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


# ── Full ODE classifier ──────────────────────────────────────────────────────

class CardiacODEClassifier(nn.Module):
    """
    Graph embeddings → Neural ODE → logits.

    v4 skip-connection:
      z0      (A_noODE) : GRU hidden state — raw motion summary
      z_final (A_ODE)   : ODE trajectory endpoint — smoothed dynamics
      z_in = concat([z0, z_final, clinical])  →  ClassifierHead

    This lets the classifier independently weight raw vs. smoothed dynamics
    rather than only seeing the ODE output.
    """

    def __init__(self, d_z: int = 64, n_classes: int = 5,
                 method: str = 'euler', n_clinical: int = 8):
        super().__init__()
        self.d_z        = d_z
        self.n_clinical = n_clinical

        self.seq_encoder = SequenceEncoder(d_z)
        self.ode_block   = NeuralODEBlock(d_z=d_z, method=method)
        self.classifier  = ClassifierHead(
            in_dim    = 2 * d_z + n_clinical,   # z0 || z_final || clinical
            n_classes = n_classes,
            dropout   = 0.5,
        )

    def forward(
        self,
        embeddings:    torch.Tensor,                    # (B, N_frames, d_z)
        times:         torch.Tensor,                    # (B, N_frames) or (N_frames,)
        clinical_feat: Optional[torch.Tensor] = None,  # (B, n_clinical) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits, z_traj).
        """
        # z0 = A_noODE : pre-ODE representation (raw motion dynamics from GRU)
        z0 = self.seq_encoder(embeddings)               # (B, d_z)

        if times.dim() == 2:
            t_span = times[0]
        else:
            t_span = times
        t_span = t_span.to(embeddings.device)
        t_span, _ = torch.sort(t_span)
        eps = 1e-5
        for i in range(1, len(t_span)):
            if t_span[i] <= t_span[i - 1]:
                t_span[i] = t_span[i - 1] + eps

        z_traj  = self.ode_block(z0, t_span)            # (N_frames, B, d_z)

        # z_final = A_ODE : post-ODE representation (smoothed continuous dynamics)
        z_final = z_traj[-1]                             # (B, d_z)

        # ── Skip connection: preserve both pre-ODE and post-ODE information ──
        # Concatenation (not addition) so the head can learn independent weights
        # for raw dynamics vs. ODE-smoothed dynamics.
        z_skip = torch.cat([z0, z_final], dim=1)         # (B, 2*d_z)

        # Append clinical features (or zero-pad if not available)
        if clinical_feat is not None and self.n_clinical > 0:
            cf   = clinical_feat.to(z_final.device)
            z_in = torch.cat([z_skip, cf], dim=1)        # (B, 2*d_z + n_clinical)
        else:
            pad  = torch.zeros(z_final.shape[0], self.n_clinical,
                               device=z_final.device)
            z_in = torch.cat([z_skip, pad], dim=1)       # (B, 2*d_z + n_clinical)

        logits = self.classifier(z_in)

        return logits, z_traj


# ── Geodesic deviation metric ────────────────────────────────────────────────

def geodesic_deviation(z_traj: torch.Tensor) -> torch.Tensor:
    N, B, d   = z_traj.shape
    z_start   = z_traj[0]
    z_end     = z_traj[-1]
    t_vals    = torch.linspace(0, 1, N, device=z_traj.device)
    geodesic  = (z_start.unsqueeze(0) * (1 - t_vals.view(N, 1, 1))
                 + z_end.unsqueeze(0) * t_vals.view(N, 1, 1))
    diff      = z_traj - geodesic
    dist      = diff.norm(dim=-1)
    arc_len   = diff.norm(dim=-1).sum(dim=0) + 1e-8
    return dist.mean(dim=0) / arc_len


# ── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N, dz = 4, 6, 64

    embeds   = torch.randn(B, N, dz, device=device)
    times    = torch.linspace(0, 1, N, device=device).unsqueeze(0).expand(B, -1)
    clin     = torch.rand(B, 8, device=device)
    labels   = torch.randint(0, 5, (B,), device=device)

    model    = CardiacODEClassifier(d_z=dz, n_classes=5, n_clinical=8).to(device)
    logits, z_traj = model(embeds, times, clinical_feat=clin)
    print(f"logits: {logits.shape}  z_traj: {z_traj.shape}")

    # Verify in_dim is 2*dz + n_clinical = 136
    expected_in = 2 * dz + 8
    actual_in   = model.classifier.head[0].in_features
    assert actual_in == expected_in, f"in_dim mismatch: {actual_in} != {expected_in}"
    print(f"ClassifierHead in_dim: {actual_in}  (2*{dz} + 8 = {expected_in}) ✓")

    loss = F.cross_entropy(logits, labels)
    loss.backward()
    gn = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"Grad norm: {gn:.4f}  (> 0 = OK)")
    print("[ode_v3.py / skip-connection] All checks passed.")
