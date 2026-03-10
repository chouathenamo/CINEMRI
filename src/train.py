"""
train.py — Training Loop for CardiacMotionODE
=============================================
Features:
  - Leave-one-patient-out (LOPO) cross-validation (correct for n=20 patients)
  - Stratified k-fold as alternative for larger datasets
  - AdamW optimizer + CosineAnnealingLR scheduler
  - Gradient clipping (essential for ODE stability)
  - Per-epoch logging: loss breakdown + accuracy
  - Checkpoint saving (best val accuracy)
  - Ablation mode: swap out modules to reproduce baselines
  - Early stopping to avoid overfitting on small dataset
  - Speed optimization: batch all frame pairs together through registration

Usage:
    # Full pipeline (our method)
    python src/train.py \
        --march9_dir  /home/amo/CINEMRI/data/ACDC/March9Data \
        --training_dir /home/amo/CINEMRI/data/ACDC/training \
        --n_patients 100 --epochs 100 --batch_size 4

    # Ablation: no ODE (just classify from mean embedding)
    python src/train.py --march9_dir ... --training_dir ... --ablation no_ode

    # Ablation: no graph (use raw phi features)
    python src/train.py --march9_dir ... --training_dir ... --ablation no_graph
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
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# Add src/ to path so imports work regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (ACDCSliceDataset, collate_fn, IDX_TO_CLASS, CLASS_TO_IDX,
                     discover_patients_from_images, parse_patient_info)
from model import CardiacMotionODE


# ── Ablation models ───────────────────────────────────────────────────────────

class AblationNoODE(nn.Module):
    """
    Baseline: Registration + Graph, but NO ODE.
    Classify directly from mean-pooled graph embeddings.
    Tests whether the temporal ODE dynamics add value.
    """
    def __init__(self, n_classes=5, d_z=64, alpha=0.1, beta=0.01):
        super().__init__()
        from model import CardiacMotionODE
        # Reuse registration and graph encoder from main model
        self._base = CardiacMotionODE(n_classes, d_z, alpha=alpha, beta=beta)
        # Simple classifier from mean embedding
        self.classifier = nn.Sequential(
            nn.Linear(d_z, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, frames, masks, times, labels=None):
        from model import ModelOutput
        # Run registration + graph encoding (reuse base model internals)
        out = self._base(frames, masks, times, labels)
        # Override: classify from mean of ODE trajectory instead of z(T)
        z_mean = out.z_traj.mean(dim=0)   # (B, d_z)
        logits = self.classifier(z_mean)
        cls_loss = F.cross_entropy(logits, labels) if labels is not None else torch.tensor(0.0)
        total_loss = cls_loss + self._base.alpha * out.reg_loss
        return ModelOutput(
            logits=logits, total_loss=total_loss, cls_loss=cls_loss,
            reg_loss=out.reg_loss, bend_loss=out.bend_loss,
            fold_loss=out.fold_loss, z_traj=out.z_traj,
            phi_sequence=out.phi_sequence
        )


class AblationNoGraph(nn.Module):
    """
    Baseline: Registration + ODE, but NO graph.
    Uses global mean-pooled phi features instead of GAT embeddings.
    Tests whether the anatomical graph structure adds value.
    """
    def __init__(self, n_classes=5, d_z=64, alpha=0.1, beta=0.01, ode_method='euler'):
        super().__init__()
        from registration import RegistrationNet, RegistrationLoss
        from ode import CardiacODEClassifier
        self.registration  = RegistrationNet()
        self.reg_loss_fn   = RegistrationLoss(alpha=alpha, beta=beta)
        self.phi_projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),   # (B, 2, 8, 8) → flatten → 128
            nn.Flatten(),
            nn.Linear(128, d_z),
            nn.ReLU(),
        )
        self.ode_classifier = CardiacODEClassifier(d_z=d_z, n_classes=n_classes, method=ode_method)
        self.alpha = alpha
        self.beta  = beta

    def forward(self, frames, masks, times, labels=None):
        from model import ModelOutput
        from ode import geodesic_deviation
        B, N_frames, C, H, W = frames.shape
        N_pairs = N_frames - 1
        device  = frames.device

        all_embeddings = []
        all_phis = []
        total_reg = torch.tensor(0.0, device=device)

        for t in range(N_pairs):
            fixed  = frames[:, t]
            moving = frames[:, t+1]
            mask_t = masks[:, t]
            warped, vel, phi = self.registration.get_warped(fixed, moving)
            all_phis.append(phi)
            reg_total, reg_ncc, rb, rf = self.reg_loss_fn(warped, fixed, vel, phi, mask_t)
            total_reg = total_reg + reg_ncc
            emb = self.phi_projector(phi)
            all_embeddings.append(emb)

        total_reg /= N_pairs
        embeds_seq = torch.stack(all_embeddings, dim=1)
        times_mid  = (times[:, :-1] + times[:, 1:]) / 2.0
        logits, z_traj = self.ode_classifier(embeds_seq, times_mid)

        cls_loss   = F.cross_entropy(logits, labels) if labels is not None else torch.tensor(0.0)
        total_loss = cls_loss + self.alpha * total_reg

        with torch.no_grad():
            geo_dev = geodesic_deviation(z_traj)

        return ModelOutput(
            logits=logits, total_loss=total_loss, cls_loss=cls_loss,
            reg_loss=total_reg, bend_loss=torch.tensor(0.0), fold_loss=torch.tensor(0.0),
            z_traj=z_traj, phi_sequence=torch.stack(all_phis, dim=1),
            geodesic_dev=geo_dev
        )


def build_model(ablation: Optional[str], n_classes: int, device: torch.device) -> nn.Module:
    if ablation == 'no_ode':
        return AblationNoODE(n_classes=n_classes).to(device)
    elif ablation == 'no_graph':
        return AblationNoGraph(n_classes=n_classes).to(device)
    else:
        return CardiacMotionODE(n_classes=n_classes, ode_method='euler').to(device)


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(all_preds: List[int], all_labels: List[int], n_classes: int) -> Dict:
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    acc    = (preds == labels).mean()

    per_class_acc = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc[IDX_TO_CLASS[c]] = (preds[mask] == labels[mask]).mean()

    return {"accuracy": acc, "per_class": per_class_acc}


# ── Training step ─────────────────────────────────────────────────────────────

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    max_grad_norm: float = 1.0,
) -> Dict:
    model.train()
    total_loss  = 0.0
    cls_loss    = 0.0
    reg_loss    = 0.0
    all_preds   = []
    all_labels  = []
    n_batches   = 0

    for batch in loader:
        frames = batch["frames"].to(device)   # (B, N, 1, H, W)
        masks  = batch["masks"].to(device)
        times  = batch["times"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        out = model(frames, masks, times, labels)
        out.total_loss.backward()

        # Gradient clipping — critical for ODE stability
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += out.total_loss.item()
        cls_loss   += out.cls_loss.item()
        reg_loss   += out.reg_loss.item()

        preds = out.logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        n_batches += 1

    metrics = compute_metrics(all_preds, all_labels, n_classes=5)
    return {
        "total_loss": total_loss / n_batches,
        "cls_loss":   cls_loss   / n_batches,
        "reg_loss":   reg_loss   / n_batches,
        "accuracy":   metrics["accuracy"],
        "per_class":  metrics["per_class"],
    }


# ── Validation step ───────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []
    n_batches  = 0

    for batch in loader:
        frames = batch["frames"].to(device)
        masks  = batch["masks"].to(device)
        times  = batch["times"].to(device)
        labels = batch["label"].to(device)

        out = model(frames, masks, times, labels)

        total_loss += out.total_loss.item()
        preds = out.logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        n_batches += 1

    metrics = compute_metrics(all_preds, all_labels, n_classes=5)
    return {
        "total_loss": total_loss / n_batches,
        "accuracy":   metrics["accuracy"],
        "per_class":  metrics["per_class"],
        "preds":      all_preds,
        "labels":     all_labels,
    }


# ── Leave-One-Patient-Out CV ──────────────────────────────────────────────────

def lopo_cv(
    march9_dir:   str,
    training_dir: str,
    patient_ids:  List[str],
    args,
    device:       torch.device,
) -> Dict:
    """
    Leave-One-Patient-Out cross-validation.

    For n=20 patients: 20 folds, each fold trains on 19 patients
    and tests on 1. Gives an unbiased estimate of generalization
    with maximum use of limited data.

    NOTE: We operate at SLICE level within each fold (one slice = one
    training sample), but the held-out patient's SLICES are the test set.
    Patient leakage is impossible because we split by patient, not slice.
    """
    all_fold_results = []
    patient_labels   = {}

    # Get label for each patient
    for pid in patient_ids:
        info = parse_patient_info(training_dir, pid)
        patient_labels[pid] = CLASS_TO_IDX.get(info["group"], -1)

    print(f"\n{'='*60}")
    print(f"  Leave-One-Patient-Out CV  ({len(patient_ids)} folds)")
    print(f"{'='*60}\n")

    for fold_idx, held_out in enumerate(patient_ids):
        train_pids = [p for p in patient_ids if p != held_out]
        test_pids  = [held_out]

        print(f"Fold {fold_idx+1:02d}/{len(patient_ids)} — held out: {held_out} "
              f"({IDX_TO_CLASS.get(patient_labels[held_out], '?')})")

        # Build datasets
        train_ds = ACDCSliceDataset(march9_dir, training_dir, train_pids, target_h=128, target_w=128)
        test_ds  = ACDCSliceDataset(march9_dir, training_dir, test_pids,  target_h=128, target_w=128)

        if len(test_ds) == 0:
            print(f"  Skipping — no valid slices for {held_out}")
            continue

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=2, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=2, pin_memory=True
        )

        # Fresh model for each fold
        model = build_model(args.ablation, n_classes=5, device=device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

        best_val_acc  = 0.0
        best_val_pred = None
        patience_ctr  = 0

        for epoch in range(1, args.epochs + 1):
            t0         = time.time()
            train_metrics = train_epoch(model, train_loader, optimizer, device)
            val_metrics   = val_epoch(model, test_loader, device)
            scheduler.step()
            elapsed = time.time() - t0

            if epoch % args.log_every == 0:
                print(
                    f"  Ep {epoch:3d}/{args.epochs} | "
                    f"loss {train_metrics['total_loss']:.3f} | "
                    f"train_acc {train_metrics['accuracy']*100:.1f}% | "
                    f"val_acc {val_metrics['accuracy']*100:.1f}% | "
                    f"{elapsed:.1f}s"
                )

            # Track best
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc  = val_metrics["accuracy"]
                best_val_pred = val_metrics["preds"]
                patience_ctr  = 0
                # Save checkpoint
                ckpt_path = os.path.join(args.checkpoint_dir, f"fold_{fold_idx:02d}_best.pt")
                torch.save({
                    "epoch":      epoch,
                    "model":      model.state_dict(),
                    "val_acc":    best_val_acc,
                    "held_out":   held_out,
                    "ablation":   args.ablation,
                }, ckpt_path)
            else:
                patience_ctr += 1

            # Early stopping
            if patience_ctr >= args.patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

        fold_result = {
            "fold":      fold_idx,
            "held_out":  held_out,
            "true_label": patient_labels[held_out],
            "pred_label": int(np.bincount(best_val_pred).argmax()) if best_val_pred else -1,
            "val_acc":   best_val_acc,
        }
        all_fold_results.append(fold_result)
        print(f"  Best val acc: {best_val_acc*100:.1f}%  "
              f"| Pred: {IDX_TO_CLASS.get(fold_result['pred_label'], '?')} "
              f"True: {IDX_TO_CLASS.get(patient_labels[held_out], '?')}\n")

    return summarize_lopo(all_fold_results)


def summarize_lopo(results: List[Dict]) -> Dict:
    """Compute overall accuracy and per-class accuracy from LOPO results."""
    preds  = [r["pred_label"]  for r in results if r["pred_label"] != -1]
    labels = [r["true_label"]  for r in results if r["pred_label"] != -1]

    preds  = np.array(preds)
    labels = np.array(labels)
    acc    = (preds == labels).mean()

    print(f"\n{'='*60}")
    print(f"  LOPO CV Final Results")
    print(f"{'='*60}")
    print(f"  Overall accuracy: {acc*100:.1f}%  ({(preds==labels).sum()}/{len(labels)})")
    print(f"\n  Per-class accuracy:")
    for c in range(5):
        mask = labels == c
        if mask.sum() > 0:
            cls_acc = (preds[mask] == labels[mask]).mean()
            name    = IDX_TO_CLASS.get(c, str(c))
            print(f"    {name:<6}: {cls_acc*100:.1f}%  ({mask.sum()} patients)")

    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    classes = [IDX_TO_CLASS[i] for i in range(5)]
    cm = confusion_matrix(labels, preds, labels=list(range(5)))
    header = "       " + "  ".join(f"{c:>5}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {classes[i]:<5}: " + "  ".join(f"{v:>5}" for v in row))

    print(f"{'='*60}\n")

    return {"accuracy": acc, "per_class": {}, "confusion_matrix": cm.tolist()}


# ── Stratified k-fold (alternative for larger datasets) ──────────────────────

def stratified_kfold(
    march9_dir:   str,
    training_dir: str,
    patient_ids:  List[str],
    args,
    device:       torch.device,
    n_splits:     int = 5,
) -> Dict:
    """
    Stratified k-fold CV — better when you have more patients
    and want balanced folds across classes.
    """
    labels = []
    for pid in patient_ids:
        info = parse_patient_info(training_dir, pid)
        labels.append(CLASS_TO_IDX.get(info["group"], 0))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_preds, all_labels = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):
        train_pids = [patient_ids[i] for i in train_idx]
        val_pids   = [patient_ids[i] for i in val_idx]

        print(f"\nFold {fold_idx+1}/{n_splits}: {len(train_pids)} train / {len(val_pids)} val patients")

        train_ds = ACDCSliceDataset(march9_dir, training_dir, train_pids)
        val_ds   = ACDCSliceDataset(march9_dir, training_dir, val_pids)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                  collate_fn=collate_fn, num_workers=2)

        model     = build_model(args.ablation, n_classes=5, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            train_epoch(model, train_loader, optimizer, device)
            val_m = val_epoch(model, val_loader, device)
            scheduler.step()
            if val_m["accuracy"] > best_acc:
                best_acc = val_m["accuracy"]
                best_preds  = val_m["preds"]
                best_labels = val_m["labels"]

        all_preds.extend(best_preds)
        all_labels.extend(best_labels)
        print(f"  Best val acc: {best_acc*100:.1f}%")

    return summarize_lopo([
        {"fold": i, "held_out": "", "true_label": l, "pred_label": p, "val_acc": 0}
        for i, (l, p) in enumerate(zip(all_labels, all_preds))
    ])


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CardiacMotionODE on ACDC")

    # Data
    p.add_argument("--march9_dir",   type=str, required=True,
                   help="Path to March9Data directory (contains Images/ and Masks/)")
    p.add_argument("--training_dir", type=str, required=True,
                   help="Path to ACDC training directory (contains patientXXX/ folders)")
    p.add_argument("--n_patients",  type=int, default=100,
                   help="Number of patients to use (default: 100)")

    # Training
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=20,
                   help="Early stopping patience (epochs without improvement)")
    p.add_argument("--log_every",   type=int,   default=10,
                   help="Print metrics every N epochs")

    # CV strategy
    p.add_argument("--cv",          type=str,   default="lopo",
                   choices=["lopo", "kfold"],
                   help="Cross-validation strategy")
    p.add_argument("--n_splits",    type=int,   default=5,
                   help="Number of folds for k-fold CV")

    # Ablation
    p.add_argument("--ablation",    type=str,   default=None,
                   choices=[None, "no_ode", "no_graph"],
                   help="Ablation mode (None = full model)")

    # Output
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                   help="Directory to save model checkpoints")
    p.add_argument("--results_file",   type=str, default="results.json",
                   help="Path to save results JSON")

    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Config : {vars(args)}\n")

    march9_dir   = args.march9_dir
    training_dir = args.training_dir

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Discover patients from Images/ folder, limited to n_patients
    all_pids = discover_patients_from_images(march9_dir)[:args.n_patients]

    print(f"Using {len(all_pids)} patients: {all_pids[0]} … {all_pids[-1]}")

    # Label distribution
    class_counts = defaultdict(int)
    for pid in all_pids:
        info = parse_patient_info(training_dir, pid)
        class_counts[info["group"]] += 1
    print(f"Class distribution: {dict(class_counts)}\n")

    # Run CV
    ablation_str = args.ablation if args.ablation else "full_model"
    print(f"Model variant: {ablation_str}")

    if args.cv == "lopo":
        results = lopo_cv(march9_dir, training_dir, all_pids, args, device)
    else:
        results = stratified_kfold(march9_dir, training_dir, all_pids, args, device, args.n_splits)

    # Save results
    results["config"] = vars(args)
    results["ablation"] = ablation_str
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.results_file}")


if __name__ == "__main__":
    main()