"""
ablation_v3.py — Ablation Study for CardiacMotionODE  (v3)
===========================================================
Runs all 4 ablation conditions against the full model in a single script.
Each condition shares the same k-fold splits and hyperparameters.

Ablation conditions
-------------------
  full_model    CardiacMotionODE v3 — registration + graph + ODE + clinical
  no_clinical   n_clinical=0  (zero pad clinical features, no EF/volume info)
  no_ode        GRU final state → classifier directly  (ODE block removed)
  no_graph      CNN pooling on phi replaces graph encoder  (no GAT)

Output
------
  results_ablation_v3.json   — all conditions, per-class F1, confusion matrices
  logs/ablation_v3.txt       — redirect stdout here during long runs

Usage
-----
  # Quick smoke test (~1-2 h):
  python ablation_v3.py \\
      --march9_dir  /path/to/March9Data \\
      --training_dir /path/to/training \\
      --n_patients 50 --epochs 20 --batch_size 4 \\
      --n_splits 3 --patience 8 \\
      --wandb_project cardiac-ode

  # Full ablation (24-36 h):
  nohup python ablation_v3.py \\
      --march9_dir  /path/to/March9Data \\
      --training_dir /path/to/training \\
      --n_patients 100 --epochs 120 --batch_size 8 \\
      --n_splits 5 --patience 15 \\
      --wandb_project cardiac-ode \\
      > logs/ablation_v3.txt 2>&1 &
  echo "PID: $!"
  tail -f logs/ablation_v3.txt
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_v3 import (
    ACDCSliceDataset, collate_fn,
    IDX_TO_CLASS, CLASS_TO_IDX,
    discover_patients_from_images, parse_patient_info,
    N_CLINICAL_FEATURES,
)
from model_v3   import CardiacMotionODE, ModelOutput
from ode_v3     import SequenceEncoder, ClassifierHead, geodesic_deviation
from graph_v3   import GraphMotionEncoder
from registration import RegistrationNet, RegistrationLoss, warp
from metrics_v3 import compute_all_metrics, print_metrics_table, metrics_to_latex, latex_table_header


# ════════════════════════════════════════════════════════════════════════════════
# Ablation model variants
# ════════════════════════════════════════════════════════════════════════════════

# ── (A) no_clinical  ─────────────────────────────────────────────────────────
# Identical to CardiacMotionODE but n_clinical=0.
# Just instantiate CardiacMotionODE(n_clinical=0).

# ── (B) no_ode  ──────────────────────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class AblationModelOutput:
    logits:       torch.Tensor
    total_loss:   torch.Tensor
    cls_loss:     torch.Tensor
    reg_loss:     torch.Tensor
    bend_loss:    torch.Tensor
    fold_loss:    torch.Tensor
    z_traj:       Optional[torch.Tensor]
    phi_sequence: torch.Tensor
    geodesic_dev: Optional[torch.Tensor] = None


class CardiacMotionNoODE(nn.Module):
    """
    Ablation: no Neural ODE.
    Pipeline: Registration → GraphEncoder → GRU → ClassifierHead
    The GRU hidden state z0 is used directly (no ODE integration).
    """

    def __init__(self, n_classes=5, d_z=64, n_verts=64, k_neighbors=6,
                 alpha=0.1, beta=0.01, n_clinical=8):
        super().__init__()
        self.alpha      = alpha
        self.beta       = beta
        self.n_clinical = n_clinical

        self.registration  = RegistrationNet()
        self.reg_loss_fn   = RegistrationLoss(alpha=alpha, beta=beta)
        self.graph_encoder = GraphMotionEncoder(n_verts=n_verts, k=k_neighbors, out_dim=d_z)
        self.seq_encoder   = SequenceEncoder(d_z)
        # No ODE block — z0 goes straight to classifier
        self.classifier    = ClassifierHead(
            in_dim=d_z + n_clinical, n_classes=n_classes, dropout=0.5)

    def forward(self, frames, masks, times, labels=None, clinical_feat=None):
        B, N_frames, C, H, W = frames.shape
        N_pairs  = N_frames - 1
        device   = frames.device

        all_embeddings  = []
        all_phis        = []
        total_reg_loss  = torch.tensor(0.0, device=device)
        total_bend_loss = torch.tensor(0.0, device=device)
        total_fold_loss = torch.tensor(0.0, device=device)

        for t in range(N_pairs):
            warped, vel_field, phi = self.registration.get_warped(
                frames[:, t], frames[:, t+1])
            all_phis.append(phi)
            reg_total, reg_ncc, reg_bend, reg_fold = self.reg_loss_fn(
                warped, frames[:, t], vel_field, phi, masks[:, t])
            total_reg_loss  += reg_ncc
            total_bend_loss += reg_bend
            total_fold_loss += reg_fold
            all_embeddings.append(self.graph_encoder(masks[:, t], phi))

        total_reg_loss  /= N_pairs
        total_bend_loss /= N_pairs
        total_fold_loss /= N_pairs

        embeddings_seq = torch.stack(all_embeddings, dim=1)  # (B, N_pairs, d_z)
        phi_sequence   = torch.stack(all_phis,       dim=1)

        # GRU → z0; skip ODE
        z0 = self.seq_encoder(embeddings_seq)  # (B, d_z)

        if clinical_feat is not None and self.n_clinical > 0:
            z_in = torch.cat([z0, clinical_feat.to(z0.device)], dim=1)
        else:
            pad  = torch.zeros(z0.shape[0], self.n_clinical, device=z0.device)
            z_in = torch.cat([z0, pad], dim=1)

        logits = self.classifier(z_in)

        if labels is not None:
            class_weight = torch.ones(5, device=device)
            class_weight[CLASS_TO_IDX["MINF"]] = 2.0
            class_weight[CLASS_TO_IDX["DCM"]]  = 1.5
            cls_loss = F.cross_entropy(logits, labels,
                                       weight=class_weight, label_smoothing=0.1)
        else:
            cls_loss = torch.tensor(0.0, device=device)

        total_loss = (cls_loss
                      + self.alpha * total_reg_loss
                      + self.beta  * (total_bend_loss + total_fold_loss))

        return AblationModelOutput(
            logits=logits, total_loss=total_loss, cls_loss=cls_loss,
            reg_loss=total_reg_loss, bend_loss=total_bend_loss,
            fold_loss=total_fold_loss, z_traj=None, phi_sequence=phi_sequence,
        )

    @torch.no_grad()
    def predict(self, frames, masks, times, clinical_feat=None):
        self.eval()
        out = self.forward(frames, masks, times, clinical_feat=clinical_feat)
        return out.logits.argmax(dim=1)


# ── (C) no_graph  ────────────────────────────────────────────────────────────

class PhiCNNEncoder(nn.Module):
    """
    Replaces the graph encoder. Encodes the deformation field phi (B, 2, H, W)
    with a small CNN + global average pool.
    Output: (B, out_dim)
    """
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2,  16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_dim, 3, stride=2, padding=1), nn.ReLU(),
        )

    def forward(self, masks: torch.Tensor, phis: torch.Tensor) -> torch.Tensor:
        """masks unused; phis: (B, 2, H, W)"""
        feat = self.net(phis)                     # (B, out_dim, H', W')
        return feat.mean(dim=[2, 3])              # global avg pool → (B, out_dim)


class CardiacMotionNoGraph(nn.Module):
    """
    Ablation: no graph encoder.
    Pipeline: Registration → CNN on phi → GRU → ODE → ClassifierHead
    """

    def __init__(self, n_classes=5, d_z=64, alpha=0.1, beta=0.01,
                 ode_method='euler', n_clinical=8):
        super().__init__()
        self.alpha      = alpha
        self.beta       = beta
        self.n_clinical = n_clinical

        self.registration  = RegistrationNet()
        self.reg_loss_fn   = RegistrationLoss(alpha=alpha, beta=beta)
        self.phi_encoder   = PhiCNNEncoder(out_dim=d_z)   # replaces graph encoder

        from ode_v3 import CardiacODEClassifier
        self.ode_classifier = CardiacODEClassifier(
            d_z=d_z, n_classes=n_classes, method=ode_method, n_clinical=n_clinical)

    def forward(self, frames, masks, times, labels=None, clinical_feat=None):
        B, N_frames, C, H, W = frames.shape
        N_pairs  = N_frames - 1
        device   = frames.device

        all_embeddings  = []
        all_phis        = []
        total_reg_loss  = torch.tensor(0.0, device=device)
        total_bend_loss = torch.tensor(0.0, device=device)
        total_fold_loss = torch.tensor(0.0, device=device)

        for t in range(N_pairs):
            warped, vel_field, phi = self.registration.get_warped(
                frames[:, t], frames[:, t+1])
            all_phis.append(phi)
            reg_total, reg_ncc, reg_bend, reg_fold = self.reg_loss_fn(
                warped, frames[:, t], vel_field, phi, masks[:, t])
            total_reg_loss  += reg_ncc
            total_bend_loss += reg_bend
            total_fold_loss += reg_fold
            # CNN on phi instead of graph
            all_embeddings.append(self.phi_encoder(masks[:, t], phi))

        total_reg_loss  /= N_pairs
        total_bend_loss /= N_pairs
        total_fold_loss /= N_pairs

        embeddings_seq = torch.stack(all_embeddings, dim=1)
        phi_sequence   = torch.stack(all_phis,       dim=1)
        times_mid      = (times[:, :-1] + times[:, 1:]) / 2.0

        logits, z_traj = self.ode_classifier(
            embeddings_seq, times_mid, clinical_feat=clinical_feat)

        if labels is not None:
            class_weight = torch.ones(5, device=device)
            class_weight[CLASS_TO_IDX["MINF"]] = 2.0
            class_weight[CLASS_TO_IDX["DCM"]]  = 1.5
            cls_loss = F.cross_entropy(logits, labels,
                                       weight=class_weight, label_smoothing=0.1)
        else:
            cls_loss = torch.tensor(0.0, device=device)

        total_loss = (cls_loss
                      + self.alpha * total_reg_loss
                      + self.beta  * (total_bend_loss + total_fold_loss))

        with torch.no_grad():
            geo_dev = geodesic_deviation(z_traj)

        return AblationModelOutput(
            logits=logits, total_loss=total_loss, cls_loss=cls_loss,
            reg_loss=total_reg_loss, bend_loss=total_bend_loss,
            fold_loss=total_fold_loss, z_traj=z_traj, phi_sequence=phi_sequence,
            geodesic_dev=geo_dev,
        )

    @torch.no_grad()
    def predict(self, frames, masks, times, clinical_feat=None):
        self.eval()
        out = self.forward(frames, masks, times, clinical_feat=clinical_feat)
        return out.logits.argmax(dim=1)


# ── Model factory ─────────────────────────────────────────────────────────────

ABLATION_CONDITIONS = ["full_model", "no_clinical", "no_ode", "no_graph"]

def build_model(condition: str, device: torch.device) -> nn.Module:
    if condition == "full_model":
        return CardiacMotionODE(
            n_classes=5, d_z=64, ode_method='euler',
            n_clinical=N_CLINICAL_FEATURES).to(device)
    elif condition == "no_clinical":
        return CardiacMotionODE(
            n_classes=5, d_z=64, ode_method='euler',
            n_clinical=0).to(device)
    elif condition == "no_ode":
        return CardiacMotionNoODE(
            n_classes=5, d_z=64, n_clinical=N_CLINICAL_FEATURES).to(device)
    elif condition == "no_graph":
        return CardiacMotionNoGraph(
            n_classes=5, d_z=64, ode_method='euler',
            n_clinical=N_CLINICAL_FEATURES).to(device)
    else:
        raise ValueError(f"Unknown ablation condition: {condition!r}")


# ════════════════════════════════════════════════════════════════════════════════
# Training helpers  (mirror train_v3.py — kept here to be self-contained)
# ════════════════════════════════════════════════════════════════════════════════

def _flush(*args):
    print(*args, flush=True)


def freeze_registration(model):
    if hasattr(model, "registration"):
        for p in model.registration.parameters():
            p.requires_grad = False


def unfreeze_registration(model, optimizer, base_lr):
    if not hasattr(model, "registration"):
        return
    for p in model.registration.parameters():
        p.requires_grad = True
    reg_params = [p for p in model.registration.parameters() if p.requires_grad]
    optimizer.add_param_group({"params": reg_params, "lr": base_lr * 0.1})
    _flush(f"    [Unfreeze] Registration added to optimizer at LR={base_lr*0.1:.2e}")


def _train_epoch(model, loader, optimizer, device, max_grad_norm=1.0):
    model.train()
    total_loss = cls_loss_sum = reg_loss_sum = 0.0
    all_preds, all_labels = [], []
    n_batches = 0

    for batch in loader:
        frames = batch["frames"].to(device)
        masks  = batch["masks"].to(device)
        times  = batch["times"].to(device)
        labels = batch["label"].to(device)
        clin   = batch["clinical_feat"].to(device)

        optimizer.zero_grad()
        out = model(frames, masks, times, labels, clinical_feat=clin)
        out.total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss   += out.total_loss.item()
        cls_loss_sum += out.cls_loss.item()
        reg_loss_sum += out.reg_loss.item()
        preds         = out.logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        n_batches += 1

    preds_a  = np.array(all_preds)
    labels_a = np.array(all_labels)
    acc      = (preds_a == labels_a).mean()
    return {
        "total_loss": total_loss   / n_batches,
        "cls_loss":   cls_loss_sum / n_batches,
        "reg_loss":   reg_loss_sum / n_batches,
        "accuracy":   acc,
    }


@torch.no_grad()
def _val_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    patient_data:      Dict[str, list] = defaultdict(list)
    patient_label_map: Dict[str, int]  = {}

    for batch in loader:
        frames = batch["frames"].to(device)
        masks  = batch["masks"].to(device)
        times  = batch["times"].to(device)
        labels = batch["label"].to(device)
        clin   = batch["clinical_feat"].to(device)

        out         = model(frames, masks, times, labels, clinical_feat=clin)
        total_loss += out.total_loss.item()
        n_batches  += 1

        logits_cpu = out.logits.detach().cpu()
        for i, meta in enumerate(batch["meta"]):
            pid = meta["patient_id"]
            w   = float(meta["n_myo_pixels"])
            patient_data[pid].append((logits_cpu[i], w))
            patient_label_map[pid] = labels[i].item()

    all_preds, all_labels_list = [], []
    for pid, data in patient_data.items():
        logit_stack = torch.stack([d[0] for d in data])
        weights     = torch.tensor([d[1] for d in data])
        weights     = weights / (weights.sum() + 1e-8)
        mean_logit  = (logit_stack * weights.unsqueeze(1)).sum(0)
        all_preds.append(int(mean_logit.argmax().item()))
        all_labels_list.append(patient_label_map[pid])

    preds_a  = np.array(all_preds)
    labels_a = np.array(all_labels_list)
    acc      = (preds_a == labels_a).mean()

    return {
        "total_loss": total_loss / max(n_batches, 1),
        "accuracy":   acc,
        "preds":      all_preds,
        "labels":     all_labels_list,
    }


# ════════════════════════════════════════════════════════════════════════════════
# Single condition: k-fold CV
# ════════════════════════════════════════════════════════════════════════════════

def run_condition(
    condition:    str,
    march9_dir:   str,
    training_dir: str,
    patient_ids:  List[str],
    labels_all:   List[int],
    fold_splits:  List[Tuple[List[int], List[int]]],
    args,
    device:       torch.device,
) -> Dict:
    """
    Runs one ablation condition across all pre-computed folds.
    Returns {accuracy, confusion_matrix, per_class_f1, ...}
    """
    _flush(f"\n{'━'*64}")
    _flush(f"  CONDITION: {condition.upper()}")
    _flush(f"{'━'*64}")

    all_preds, all_true = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        train_pids = [patient_ids[i] for i in train_idx]
        val_pids   = [patient_ids[i] for i in val_idx]

        _flush(f"\n  Fold {fold_idx+1}/{len(fold_splits)}: "
               f"{len(train_pids)} train / {len(val_pids)} val")

        train_ds = ACDCSliceDataset(march9_dir, training_dir, train_pids, augment=True)
        val_ds   = ACDCSliceDataset(march9_dir, training_dir, val_pids,   augment=False)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  collate_fn=collate_fn,
                                  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate_fn,
                                  num_workers=2, pin_memory=True)

        model = build_model(condition, device)
        freeze_registration(model)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)

        use_wandb = WANDB_AVAILABLE and getattr(args, "wandb_project", None)
        if use_wandb:
            run_name = f"ablation_{condition}_fold{fold_idx+1}"
            wandb.init(project=args.wandb_project, name=run_name,
                       config={**vars(args), "condition": condition},
                       reinit=True, tags=["ablation_v3", condition])

        best_val_acc  = 0.0
        best_val_loss = float("inf")
        best_preds    = []
        best_labels   = []
        patience_ctr  = 0
        reg_unfrozen  = False

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            if not reg_unfrozen and epoch > args.freeze_reg_epochs:
                unfreeze_registration(model, optimizer, args.lr)
                reg_unfrozen = True

            train_m = _train_epoch(model, train_loader, optimizer, device)
            val_m   = _val_epoch(model, val_loader, device)
            scheduler.step()

            _flush(
                f"    [Fold {fold_idx+1} | Ep {epoch:3d}/{args.epochs}] "
                f"train_acc={train_m['accuracy']*100:.1f}%  "
                f"val_acc={val_m['accuracy']*100:.1f}%  "
                f"val_loss={val_m['total_loss']:.3f}  "
                f"({time.time()-t0:.0f}s)"
            )

            if use_wandb:
                wandb.log({
                    "epoch":          epoch,
                    "train/loss":     train_m["total_loss"],
                    "train/cls_loss": train_m["cls_loss"],
                    "train/acc":      train_m["accuracy"],
                    "val/acc":        val_m["accuracy"],
                    "val/loss":       val_m["total_loss"],
                    "lr":             scheduler.get_last_lr()[0],
                })

            if val_m["accuracy"] > best_val_acc:
                best_val_acc  = val_m["accuracy"]
                best_preds    = val_m["preds"]
                best_labels   = val_m["labels"]

            if val_m["total_loss"] < best_val_loss:
                best_val_loss = val_m["total_loss"]
                patience_ctr  = 0
            else:
                patience_ctr += 1

            if patience_ctr >= args.patience:
                _flush(f"    Early stopping at epoch {epoch}")
                break

        if use_wandb:
            wandb.finish()

        _flush(f"  Best val acc: {best_val_acc*100:.1f}%")
        all_preds.extend(best_preds)
        all_true.extend(best_labels)

    # ── Compute full metrics for this condition ───────────────────────────────
    metrics = compute_all_metrics(all_preds, all_true)
    print_metrics_table(metrics, title=f"Ablation — {condition}")

    return {
        "condition":        condition,
        "accuracy":         metrics["accuracy"],
        "acc_4class":       metrics["acc_4class"],
        "macro_f1":         metrics["macro_f1"],
        "weighted_f1":      metrics["weighted_f1"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "per_class": {
            name: {k: v for k, v in vals.items() if k not in ("tp","fp","fn")}
            for name, vals in metrics["per_class"].items()
        },
    }


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Ablation study for CardiacMotionODE v3")
    p.add_argument("--march9_dir",        type=str, required=True)
    p.add_argument("--training_dir",      type=str, required=True)
    p.add_argument("--n_patients",        type=int, default=100)
    p.add_argument("--epochs",            type=int, default=120)
    p.add_argument("--batch_size",        type=int, default=8)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--patience",          type=int, default=15)
    p.add_argument("--freeze_reg_epochs", type=int, default=20)
    p.add_argument("--n_splits",          type=int, default=5)
    p.add_argument("--conditions",        type=str, nargs="+",
                   default=ABLATION_CONDITIONS,
                   help="Subset of conditions to run (default: all 4)")
    p.add_argument("--results_file",      type=str, default="results_ablation_v3.json")
    p.add_argument("--wandb_project",     type=str, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _flush(f"Device : {device}")
    _flush(f"Config : {vars(args)}\n")

    os.makedirs("logs", exist_ok=True)

    # ── Discover patients and compute shared fold splits ─────────────────────
    all_pids = discover_patients_from_images(args.march9_dir)[:args.n_patients]
    _flush(f"Patients: {len(all_pids)}  ({all_pids[0]} … {all_pids[-1]})")

    labels_all = []
    class_counts = defaultdict(int)
    for pid in all_pids:
        info = parse_patient_info(args.training_dir, pid)
        idx  = CLASS_TO_IDX.get(info["group"], 0)
        labels_all.append(idx)
        class_counts[info["group"]] += 1
    _flush(f"Class distribution: {dict(class_counts)}\n")

    skf          = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    fold_splits  = list(skf.split(all_pids, labels_all))
    _flush(f"Fold sizes: "
           + "  ".join(f"fold{i+1}: {len(v)} val" for i, (_, v) in enumerate(fold_splits)))

    # ── Run each condition ────────────────────────────────────────────────────
    all_results = {}
    for condition in args.conditions:
        result = run_condition(
            condition, args.march9_dir, args.training_dir,
            all_pids, labels_all, fold_splits, args, device)
        all_results[condition] = result

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "conditions": all_results,
        "config":     vars(args),
    }
    with open(args.results_file, "w") as f:
        json.dump(output, f, indent=2)
    _flush(f"\nResults saved → {args.results_file}")

    # ── Summary table ─────────────────────────────────────────────────────────
    _flush(f"\n{'═'*64}")
    _flush("  ABLATION SUMMARY TABLE")
    _flush(f"{'═'*64}")
    header = (f"  {'Condition':<18}  {'Acc(5)':>7}  {'Acc(4)':>7}  "
              f"{'MacroF1':>8}  " +
              "  ".join(f"{n:>6}" for n in ["NOR","DCM","HCM","MINF","RV"]) +
              "  F1")
    _flush(header)
    _flush("  " + "─" * (len(header) - 2))
    for cond, r in all_results.items():
        pc   = r["per_class"]
        f1s  = "  ".join(f"{pc[n]['f1']*100:>5.1f}%" for n in ["NOR","DCM","HCM","MINF","RV"])
        _flush(f"  {cond:<18}  {r['accuracy']*100:>6.1f}%  "
               f"{r['acc_4class']*100:>6.1f}%  "
               f"{r['macro_f1']*100:>7.1f}%  {f1s}")

    # LaTeX table
    _flush(f"\n  LaTeX (paste into Table 2):")
    _flush("  " + latex_table_header())
    for cond, r in all_results.items():
        _flush("  " + metrics_to_latex(
            {**r, "confusion_matrix": np.array(r["confusion_matrix"])},
            model_name=cond.replace("_", "\\_"),
        ))
    _flush("  \\bottomrule\n  \\end{tabular}")


if __name__ == "__main__":
    main()
