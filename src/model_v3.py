"""
model_v3.py — Full Cardiac Motion ODE Pipeline  (v3)
=====================================================
Changes from v2:
  - CardiacMotionODE accepts clinical_feat (B, n_clinical) in forward()
    and passes it to CardiacODEClassifier → ClassifierHead
  - n_clinical=8 parameter (matches dataset_v3.N_CLINICAL_FEATURES)
  - Class-weighted cross-entropy: MINF x2.0, DCM x1.5 (hardest confusable pair)
  - Ablation models updated accordingly

Import names:
    from model_v3 import CardiacMotionODE, PatientLevelWrapper, ModelOutput
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from registration import RegistrationNet, RegistrationLoss, warp
from graph_v3   import GraphMotionEncoder
from ode_v3     import CardiacODEClassifier, geodesic_deviation
from dataset_v3 import CLASS_TO_IDX


# ── Output container ──────────────────────────────────────────────────────────

@dataclass
class ModelOutput:
    logits:       torch.Tensor
    total_loss:   torch.Tensor
    cls_loss:     torch.Tensor
    reg_loss:     torch.Tensor
    bend_loss:    torch.Tensor
    fold_loss:    torch.Tensor
    z_traj:       torch.Tensor
    phi_sequence: torch.Tensor
    geodesic_dev: Optional[torch.Tensor] = None


# ── Main model ────────────────────────────────────────────────────────────────

class CardiacMotionODE(nn.Module):
    """
    End-to-end cardiac motion ODE model for ACDC classification.

    v3 additions:
      - n_clinical : number of clinical feature scalars (default 8)
                     These are concatenated to z(T) before the classifier.
                     Set to 0 to disable (ablation).
    """

    def __init__(
        self,
        n_classes:   int   = 5,
        d_z:         int   = 64,
        n_verts:     int   = 64,
        k_neighbors: int   = 6,
        alpha:       float = 0.1,
        beta:        float = 0.01,
        ode_method:  str   = 'euler',
        n_clinical:  int   = 8,
    ):
        super().__init__()
        self.alpha      = alpha
        self.beta       = beta
        self.n_clinical = n_clinical

        self.registration   = RegistrationNet()
        self.reg_loss_fn    = RegistrationLoss(alpha=alpha, beta=beta)
        self.graph_encoder  = GraphMotionEncoder(
            n_verts=n_verts, k=k_neighbors, out_dim=d_z)
        self.ode_classifier = CardiacODEClassifier(
            d_z=d_z, n_classes=n_classes,
            method=ode_method, n_clinical=n_clinical,
        )

    def forward(
        self,
        frames:        torch.Tensor,              # (B, N_frames, 1, H, W)
        masks:         torch.Tensor,              # (B, N_frames, 1, H, W)
        times:         torch.Tensor,              # (B, N_frames)
        labels:        Optional[torch.Tensor] = None,   # (B,)
        clinical_feat: Optional[torch.Tensor] = None,   # (B, n_clinical)
    ) -> ModelOutput:

        B, N_frames, C, H, W = frames.shape
        N_pairs  = N_frames - 1
        device   = frames.device

        all_embeddings  = []
        all_phis        = []
        total_reg_loss  = torch.tensor(0.0, device=device)
        total_bend_loss = torch.tensor(0.0, device=device)
        total_fold_loss = torch.tensor(0.0, device=device)

        for t in range(N_pairs):
            frame_fixed  = frames[:, t,   :, :, :]
            frame_moving = frames[:, t+1, :, :, :]
            mask_t       = masks[:,  t,   :, :, :]

            warped, vel_field, phi = self.registration.get_warped(
                frame_fixed, frame_moving)
            all_phis.append(phi)

            reg_total, reg_ncc, reg_bend, reg_fold = self.reg_loss_fn(
                warped, frame_fixed, vel_field, phi, mask_t)
            total_reg_loss  = total_reg_loss  + reg_ncc
            total_bend_loss = total_bend_loss + reg_bend
            total_fold_loss = total_fold_loss + reg_fold

            embedding = self.graph_encoder(mask_t, phi)
            all_embeddings.append(embedding)

        total_reg_loss  = total_reg_loss  / N_pairs
        total_bend_loss = total_bend_loss / N_pairs
        total_fold_loss = total_fold_loss / N_pairs

        embeddings_seq = torch.stack(all_embeddings, dim=1)  # (B, N_pairs, d_z)
        phi_sequence   = torch.stack(all_phis,       dim=1)  # (B, N_pairs, 2, H, W)

        times_mid = (times[:, :-1] + times[:, 1:]) / 2.0    # (B, N_pairs)

        # Clinical features passed to ODE classifier for fusion at z(T)
        logits, z_traj = self.ode_classifier(
            embeddings_seq, times_mid, clinical_feat=clinical_feat)

        if labels is not None:
            # Up-weight the two hardest / most clinically important classes.
            # MINF is easiest to miss (looks like DCM volumetrically) -> x2.0
            # DCM is the other half of that confusable pair          -> x1.5
            class_weight = torch.ones(5, device=device)
            class_weight[CLASS_TO_IDX["MINF"]] = 2.0
            class_weight[CLASS_TO_IDX["DCM"]]  = 1.5
            cls_loss = F.cross_entropy(
                logits, labels,
                weight=class_weight,
                label_smoothing=0.1,
            )
        else:
            cls_loss = torch.tensor(0.0, device=device)

        total_loss = (
            cls_loss
            + self.alpha * total_reg_loss
            + self.beta  * (total_bend_loss + total_fold_loss)
        )

        with torch.no_grad():
            geo_dev = geodesic_deviation(z_traj)

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
    def predict(self, frames, masks, times, clinical_feat=None):
        self.eval()
        out = self.forward(frames, masks, times,
                           labels=None, clinical_feat=clinical_feat)
        return out.logits.argmax(dim=1)

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "registration":   n(self.registration),
            "graph_encoder":  n(self.graph_encoder),
            "ode_classifier": n(self.ode_classifier),
            "total":          n(self),
        }


# ── Patient-level wrapper ─────────────────────────────────────────────────────

class PatientLevelWrapper(nn.Module):
    """Aggregates per-slice logits (mean-pool) for patient-level prediction."""

    def __init__(self, model: CardiacMotionODE):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def predict_patient(
        self,
        frames:        torch.Tensor,          # (D, N_frames, 1, H, W)
        masks:         torch.Tensor,
        times:         torch.Tensor,
        clinical_feat: Optional[torch.Tensor] = None,  # (1, n_clinical) or (D, n_clinical)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        D = frames.shape[0]
        if times.dim() == 1:
            times = times.unsqueeze(0).expand(D, -1)
        if clinical_feat is not None and clinical_feat.dim() == 1:
            clinical_feat = clinical_feat.unsqueeze(0).expand(D, -1)

        out        = self.model.forward(frames, masks, times,
                                        labels=None, clinical_feat=clinical_feat)
        mean_logit = out.logits.mean(dim=0)
        pred_class = mean_logit.argmax()
        return pred_class, out.logits


# ── Sanity check ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time as tm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = CardiacMotionODE(n_classes=5, d_z=64, n_clinical=8, ode_method='euler').to(device)
    params = model.count_parameters()
    for k, v in params.items():
        print(f"  {k:<20}: {v:>10,}")

    B, N, H, W = 2, 6, 128, 128
    frames  = torch.rand(B, N, 1, H, W, device=device)
    masks   = torch.zeros(B, N, 1, H, W, device=device)
    import numpy as np
    cx, cy  = W // 2, H // 2
    for i in range(H):
        for j in range(W):
            r = np.sqrt((i - cy)**2 + (j - cx)**2)
            if 20 < r < 35:
                masks[:, :, :, i, j] = 1.0

    times   = torch.linspace(0, 1, N, device=device).unsqueeze(0).expand(B, -1)
    labels  = torch.tensor([1, 3], device=device)
    clin    = torch.rand(B, 8, device=device)

    t0  = tm.time()
    out = model(frames, masks, times, labels, clin)
    t1  = tm.time()

    print(f"\nForward: {(t1-t0)*1000:.0f} ms")
    print(f"logits : {out.logits.shape}")
    print(f"loss   : total={out.total_loss.item():.4f}  "
          f"cls={out.cls_loss.item():.4f}  reg={out.reg_loss.item():.4f}")

    out.total_loss.backward()
    gns = {}
    for name, mod in [("registration", model.registration),
                      ("graph_encoder", model.graph_encoder),
                      ("ode_classifier", model.ode_classifier)]:
        gns[name] = sum(p.grad.norm().item() for p in mod.parameters() if p.grad is not None)
        print(f"grad [{name}]: {gns[name]:.4f}")

    print(f"\nGradients through all modules: {all(v > 0 for v in gns.values())} ✓")
    print("[model_v3.py] All checks passed.")
