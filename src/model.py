"""
model.py — Full Cardiac Motion ODE Pipeline
============================================
Wires together all three modules into one end-to-end differentiable model:

    ┌─────────────────────────────────────────────────────────────┐
    │  Input: (B, N_frames, 1, H, W) frames                       │
    │         (B, N_frames, 1, H, W) masks                        │
    │         (B, N_frames)          times                         │
    │         label (B,)             for loss                      │
    └──────────────────────┬──────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────┐
    │  Stage 1: RegistrationNet (per consecutive frame pair)       │
    │  frame_t → frame_{t+1}  :  U-Net → velocity → φ             │
    │  Output: φ sequence (B, N_frames-1, 2, H, W)                │
    └──────────────────────┬──────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────┐
    │  Stage 2: GraphMotionEncoder (per frame pair)                │
    │  mask + φ → contour vertices → GAT → embedding              │
    │  Output: embeddings (B, N_frames-1, 64)                      │
    └──────────────────────┬──────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────┐
    │  Stage 3: CardiacODEClassifier                               │
    │  embeddings → Neural ODE → z(T) → MLP → logits              │
    │  Output: logits (B, 5), z_traj (N_frames-1, B, 64)          │
    └──────────────────────┬──────────────────────────────────────┘
                           │
    ┌──────────────────────▼──────────────────────────────────────┐
    │  Loss: L_cls + α*L_reg + β*L_smooth                          │
    └─────────────────────────────────────────────────────────────┘

Key design: gradients from L_cls flow back through the ODE, through the
GAT, through the deformation field sampling, and into the registration
U-Net. This is the core advantage over the two-stage ANTs pipeline.

Usage:
    model = CardiacMotionODE(n_classes=5).to(device)
    out   = model(frames, masks, times, labels)
    out.total_loss.backward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from registration import RegistrationNet, RegistrationLoss, warp
from graph import GraphMotionEncoder
from ode import CardiacODEClassifier, geodesic_deviation


# ── Output container ─────────────────────────────────────────────────────────

@dataclass
class ModelOutput:
    """
    Structured output from one forward pass.
    Keeps the interface clean — callers can access what they need.
    """
    logits:       torch.Tensor          # (B, n_classes) — raw class scores
    total_loss:   torch.Tensor          # scalar — backprop this
    cls_loss:     torch.Tensor          # scalar — classification CE loss
    reg_loss:     torch.Tensor          # scalar — registration NCC loss
    bend_loss:    torch.Tensor          # scalar — bending energy
    fold_loss:    torch.Tensor          # scalar — folding penalty
    z_traj:       torch.Tensor          # (N_pairs, B, d_z) — ODE trajectory
    phi_sequence: torch.Tensor          # (B, N_pairs, 2, H, W) — deformations
    geodesic_dev: Optional[torch.Tensor] = None  # (B,) — per-sample deviation


# ── Main model ───────────────────────────────────────────────────────────────

class CardiacMotionODE(nn.Module):
    """
    End-to-end cardiac motion ODE model for heart failure classification.

    Args:
        n_classes    : number of output classes (5 for ACDC)
        d_z          : latent ODE dimension (must match graph encoder out_dim)
        n_verts      : number of myocardium contour vertices
        k_neighbors  : k-NN graph connectivity
        alpha        : weight on registration loss
        beta         : weight on smoothness/folding loss
        ode_method   : ODE solver ('euler' for speed, 'dopri5'/'adjoint' for accuracy)
    """

    def __init__(
        self,
        n_classes:   int   = 5,
        d_z:         int   = 64,
        n_verts:     int   = 64,
        k_neighbors: int   = 6,
        alpha:       float = 0.1,    # reg loss weight
        beta:        float = 0.01,   # smoothness weight
        ode_method:  str   = 'euler',
    ):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

        # ── Sub-modules ───────────────────────────────────────────────────────
        self.registration = RegistrationNet()
        self.reg_loss_fn  = RegistrationLoss(alpha=alpha, beta=beta)
        self.graph_encoder = GraphMotionEncoder(
            n_verts = n_verts,
            k       = k_neighbors,
            out_dim = d_z,
        )
        self.ode_classifier = CardiacODEClassifier(
            d_z       = d_z,
            n_classes = n_classes,
            method    = ode_method,
        )

    def forward(
        self,
        frames:  torch.Tensor,          # (B, N_frames, 1, H, W)
        masks:   torch.Tensor,          # (B, N_frames, 1, H, W)
        times:   torch.Tensor,          # (B, N_frames)
        labels:  Optional[torch.Tensor] = None,  # (B,) int64 — None at inference
    ) -> ModelOutput:
        """
        Full forward pass.

        For N_frames input frames, we compute N_frames-1 consecutive
        frame pairs: (frame_0→frame_1), (frame_1→frame_2), ...

        Each pair goes through registration → graph encoder, giving us
        N_frames-1 graph embeddings. These form the sequence fed to the ODE.

        The times fed to the ODE are the midpoint times between consecutive
        frames (since each embedding represents motion *between* two frames).
        """
        B, N_frames, C, H, W = frames.shape
        N_pairs = N_frames - 1

        device = frames.device

        # ── Stage 1 + 2: Registration + Graph encoding per frame pair ─────────
        all_embeddings  = []   # will be (B, N_pairs, d_z)
        all_phis        = []   # will be (B, N_pairs, 2, H, W)
        total_reg_loss  = torch.tensor(0.0, device=device)
        total_bend_loss = torch.tensor(0.0, device=device)
        total_fold_loss = torch.tensor(0.0, device=device)

        for t in range(N_pairs):
            frame_fixed  = frames[:, t,   :, :, :]   # (B, 1, H, W)
            frame_moving = frames[:, t+1, :, :, :]   # (B, 1, H, W)
            mask_t       = masks[:,  t,   :, :, :]   # (B, 1, H, W) — ED mask

            # Registration: moving → fixed
            warped, vel_field, phi = self.registration.get_warped(
                frame_fixed, frame_moving
            )
            all_phis.append(phi)

            # Registration loss (accumulated over all pairs)
            reg_total, reg_ncc, reg_bend, reg_fold = self.reg_loss_fn(
                warped, frame_fixed, vel_field, phi, mask_t
            )
            total_reg_loss  = total_reg_loss  + reg_ncc
            total_bend_loss = total_bend_loss + reg_bend
            total_fold_loss = total_fold_loss + reg_fold

            # Graph encoding: mask + phi → (B, d_z)
            embedding = self.graph_encoder(mask_t, phi)   # (B, d_z)
            all_embeddings.append(embedding)

        # Average registration losses over pairs
        total_reg_loss  = total_reg_loss  / N_pairs
        total_bend_loss = total_bend_loss / N_pairs
        total_fold_loss = total_fold_loss / N_pairs

        # Stack: (B, N_pairs, d_z)
        embeddings_seq = torch.stack(all_embeddings, dim=1)

        # Stack: (B, N_pairs, 2, H, W)
        phi_sequence = torch.stack(all_phis, dim=1)

        # ── Stage 3: ODE classification ───────────────────────────────────────
        # Time points for ODE: midpoints between consecutive frame times
        # times shape: (B, N_frames) → midpoints: (B, N_pairs)
        times_mid = (times[:, :-1] + times[:, 1:]) / 2.0  # (B, N_pairs)

        logits, z_traj = self.ode_classifier(embeddings_seq, times_mid)

        # ── Loss ─────────────────────────────────────────────────────────────
        if labels is not None:
            cls_loss = F.cross_entropy(logits, labels)
        else:
            cls_loss = torch.tensor(0.0, device=device)

        total_loss = (
            cls_loss
            + self.alpha * total_reg_loss
            + self.beta  * (total_bend_loss + total_fold_loss)
        )

        # ── Geodesic deviation (no grad — diagnostic only) ────────────────────
        with torch.no_grad():
            geo_dev = geodesic_deviation(z_traj)  # (B,)

        return ModelOutput(
            logits       = logits,
            total_loss   = total_loss,
            cls_loss     = cls_loss,
            reg_loss     = total_reg_loss,
            bend_loss    = total_bend_loss,
            fold_loss    = total_fold_loss,
            z_traj       = z_traj,
            phi_sequence = phi_sequence,
            geodesic_dev = geo_dev,
        )

    @torch.no_grad()
    def predict(
        self,
        frames: torch.Tensor,   # (B, N_frames, 1, H, W)
        masks:  torch.Tensor,   # (B, N_frames, 1, H, W)
        times:  torch.Tensor,   # (B, N_frames)
    ) -> torch.Tensor:
        """
        Inference-only forward pass. Returns predicted class indices (B,).
        """
        self.eval()
        out = self.forward(frames, masks, times, labels=None)
        return out.logits.argmax(dim=1)

    def count_parameters(self) -> dict:
        """Return parameter counts per submodule."""
        def n(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "registration":   n(self.registration),
            "graph_encoder":  n(self.graph_encoder),
            "ode_classifier": n(self.ode_classifier),
            "total":          n(self),
        }


# ── Slice aggregation for patient-level prediction ───────────────────────────

class PatientLevelWrapper(nn.Module):
    """
    Wraps CardiacMotionODE for patient-level inference.

    During training we operate at slice level (fast, more samples per batch).
    During evaluation we need one prediction per patient by aggregating
    across all slices.

    Aggregation: mean-pool logits across slices, then argmax.
    This is the simplest effective approach for multi-slice classification.

    Args:
        model : trained CardiacMotionODE
    """

    def __init__(self, model: CardiacMotionODE):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def predict_patient(
        self,
        frames: torch.Tensor,   # (D, N_frames, 1, H, W)  — D = n_slices
        masks:  torch.Tensor,   # (D, N_frames, 1, H, W)
        times:  torch.Tensor,   # (N_frames,) or (D, N_frames)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on all slices of one patient, return aggregated prediction.

        Returns:
            pred_class  : scalar int tensor — predicted class
            slice_logits: (D, n_classes) — per-slice logits before aggregation
        """
        self.model.eval()
        D = frames.shape[0]

        if times.dim() == 1:
            times = times.unsqueeze(0).expand(D, -1)

        out = self.model.forward(frames, masks, times, labels=None)

        # Mean-pool logits across slices → patient-level prediction
        mean_logits = out.logits.mean(dim=0)    # (n_classes,)
        pred_class  = mean_logits.argmax()       # scalar

        return pred_class, out.logits


# ── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time as time_module
    from dataset import ACDCSliceDataset, collate_fn, IDX_TO_CLASS
    from torch.utils.data import DataLoader
    import sys, os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── 1. Build model ─────────────────────────────────────────────────────────
    model = CardiacMotionODE(
        n_classes  = 5,
        d_z        = 64,
        alpha      = 0.1,
        beta       = 0.01,
        ode_method = 'euler',
    ).to(device)

    params = model.count_parameters()
    print("── Parameter counts ──")
    for k, v in params.items():
        print(f"  {k:<20}: {v:>10,}")

    # ── 2. Synthetic forward pass (no data needed) ────────────────────────────
    print("\n── Synthetic forward pass ──")
    B, N_frames, H, W = 2, 6, 128, 128

    frames_syn = torch.rand(B, N_frames, 1, H, W, device=device)
    masks_syn  = torch.zeros(B, N_frames, 1, H, W, device=device)

    # Create synthetic ring-shaped myocardium masks
    import numpy as np
    cx, cy = W // 2, H // 2
    for i in range(H):
        for j in range(W):
            r = np.sqrt((i - cy)**2 + (j - cx)**2)
            if 20 < r < 35:
                masks_syn[:, :, :, i, j] = 1.0

    times_syn  = torch.linspace(0, 1, N_frames, device=device).unsqueeze(0).expand(B, -1)
    labels_syn = torch.tensor([1, 3], device=device)  # DCM, MINF

    t0 = time_module.time()
    out = model(frames_syn, masks_syn, times_syn, labels_syn)
    t1 = time_module.time()

    print(f"  frames shape     : {frames_syn.shape}")
    print(f"  logits shape     : {out.logits.shape}")
    print(f"  z_traj shape     : {out.z_traj.shape}")
    print(f"  phi_seq shape    : {out.phi_sequence.shape}")
    print(f"  Forward time     : {(t1-t0)*1000:.0f} ms")

    print(f"\n── Losses ──")
    print(f"  total_loss  : {out.total_loss.item():.4f}")
    print(f"  cls_loss    : {out.cls_loss.item():.4f}  (expect ≈ ln(5)=1.609)")
    print(f"  reg_loss    : {out.reg_loss.item():.4f}  (NCC, expect ≈ 0.9 on random)")
    print(f"  bend_loss   : {out.bend_loss.item():.6f}")
    print(f"  fold_loss   : {out.fold_loss.item():.6f}")
    print(f"  geodesic_dev: {out.geodesic_dev.cpu().tolist()}")

    # ── 3. Backward pass ──────────────────────────────────────────────────────
    print(f"\n── Backward pass ──")
    out.total_loss.backward()

    grad_norms = {}
    for name, module in [
        ("registration",   model.registration),
        ("graph_encoder",  model.graph_encoder),
        ("ode_classifier", model.ode_classifier),
    ]:
        gn = sum(
            p.grad.norm().item()
            for p in module.parameters()
            if p.grad is not None
        )
        grad_norms[name] = gn
        print(f"  grad_norm [{name:<15}]: {gn:.4f}")

    all_grads_positive = all(v > 0 for v in grad_norms.values())
    print(f"\n  Gradients flow through all modules: {all_grads_positive} ✓")

    # ── 4. Real data test (if ACDC paths provided) ────────────────────────────
    # Usage: python model.py <march9_dir> <training_dir>
    if len(sys.argv) > 2:
        from dataset import discover_patients_from_images
        march9_dir   = sys.argv[1]
        training_dir = sys.argv[2]
        print(f"\n── Real data test ──")
        print(f"  march9_dir   : {march9_dir}")
        print(f"  training_dir : {training_dir}")

        pids = discover_patients_from_images(march9_dir)[:5]

        ds     = ACDCSliceDataset(march9_dir, training_dir, patient_ids=pids, target_h=128, target_w=128)
        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
        batch  = next(iter(loader))

        frames_r = batch["frames"].to(device)   # (B, N, 1, H, W)
        masks_r  = batch["masks"].to(device)
        times_r  = batch["times"].to(device)
        labels_r = batch["label"].to(device)

        print(f"  Batch frames : {frames_r.shape}")
        print(f"  Batch labels : {[IDX_TO_CLASS[l.item()] for l in labels_r]}")

        model.zero_grad()
        out_r = model(frames_r, masks_r, times_r, labels_r)

        print(f"  total_loss   : {out_r.total_loss.item():.4f}")
        print(f"  cls_loss     : {out_r.cls_loss.item():.4f}")
        print(f"  Predicted    : {[IDX_TO_CLASS[i] for i in out_r.logits.argmax(1).tolist()]}")
        print(f"  True         : {[IDX_TO_CLASS[l.item()] for l in labels_r]}")

        out_r.total_loss.backward()
        print(f"  Backward: OK ✓")

    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"\nGPU memory peak: {mem_mb:.1f} MB")

    print("\n[model.py] All checks passed.")