"""
model_v3_idea3.py — identical to model_v3_residual.py with one change:
    from ode_v3_idea3 import ...   (was: from ode_v3_residual import ...)
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
from graph_v3         import GraphMotionEncoder
from ode_v3_idea3     import CardiacODEClassifier, geodesic_deviation   # ← only change
from dataset_v3       import CLASS_TO_IDX


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


class CardiacMotionODE(nn.Module):
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
        frames:        torch.Tensor,
        masks:         torch.Tensor,
        times:         torch.Tensor,
        labels:        Optional[torch.Tensor] = None,
        clinical_feat: Optional[torch.Tensor] = None,
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

        embeddings_seq = torch.stack(all_embeddings, dim=1)
        phi_sequence   = torch.stack(all_phis,       dim=1)
        times_mid      = (times[:, :-1] + times[:, 1:]) / 2.0

        logits, z_traj = self.ode_classifier(
            embeddings_seq, times_mid, clinical_feat=clinical_feat)

        if labels is not None:
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


class PatientLevelWrapper(nn.Module):
    def __init__(self, model: CardiacMotionODE):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def predict_patient(
        self,
        frames:        torch.Tensor,
        masks:         torch.Tensor,
        times:         torch.Tensor,
        clinical_feat: Optional[torch.Tensor] = None,
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
