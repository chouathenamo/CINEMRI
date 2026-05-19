"""
train_v3.py — Training Loop for CardiacMotionODE  (v3)
======================================================
Changes from v2:
  - weight_decay : 1e-5 → 5e-4
  - label_smoothing 0.1  (inside model_v3.py forward)
  - Early stopping now monitors val_loss (not val_acc) — more stable
    signal; best predictions for final confusion matrix still tracked
    by val_acc
  - Freeze registration net for first --freeze_reg_epochs (default 20)
    epochs, then add it back at 10% of base LR
  - Training datasets built with augment=True (flip, rotate, intensity)
  - val_epoch uses myo-pixel-weighted slice aggregation
  - clinical_feat passed through train and val loops
  - Per-epoch heartbeat print so you know the process is alive
  - patience default lowered to 15

Usage — quick smoke test (≈30-45 min):
    nohup python train_v3.py \\
        --march9_dir  /home/amo/CINEMRI/data/ACDC/March9Data \\
        --training_dir /home/amo/CINEMRI/data/ACDC/training \\
        --n_patients 50 --epochs 20 --batch_size 4 \\
        --n_splits 3 --patience 8 --log_every 1 --cv kfold \\
        --wandb_project cardiac-ode --wandb_run_name quick_v3 \\
        > logs/quick_v3.txt 2>&1 &
    echo "PID: $!"
    tail -f logs/quick_v3.txt

Full run (8-10 h):
    nohup python train_v3.py \\
        --march9_dir  /home/amo/CINEMRI/data/ACDC/March9Data \\
        --training_dir /home/amo/CINEMRI/data/ACDC/training \\
        --n_patients 100 --epochs 120 --batch_size 8 \\
        --n_splits 5 --patience 15 --log_every 5 --cv kfold \\
        --wandb_project cardiac-ode --wandb_run_name full_v3 \\
        > logs/full_v3.txt 2>&1 &
    echo "PID: $!"
    tail -f logs/full_v3.txt
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
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_v3 import (ACDCSliceDataset, collate_fn,
                         IDX_TO_CLASS, CLASS_TO_IDX,
                         discover_patients_from_images, parse_patient_info,
                         N_CLINICAL_FEATURES)
from model_v3 import CardiacMotionODE, ModelOutput


# ── Helpers ──────────────────────────────────────────────────────────────────

def _flush(*args):
    print(*args, flush=True)


def compute_metrics(all_preds, all_labels, n_classes=5):
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    acc    = (preds == labels).mean()
    per_class = {}
    for c in range(n_classes):
        m = labels == c
        if m.sum() > 0:
            per_class[IDX_TO_CLASS[c]] = (preds[m] == labels[m]).mean()
    return {"accuracy": acc, "per_class": per_class}


# ── Freeze / unfreeze registration ──────────────────────────────────────────

def freeze_registration(model):
    for p in model.registration.parameters():
        p.requires_grad = False

def unfreeze_registration(model, optimizer, base_lr):
    """Enable registration gradients and add them to optimizer at 10% LR."""
    for p in model.registration.parameters():
        p.requires_grad = True
    reg_params = [p for p in model.registration.parameters() if p.requires_grad]
    optimizer.add_param_group({"params": reg_params, "lr": base_lr * 0.1})
    _flush(f"  [Unfreeze] Registration net added to optimizer at LR={base_lr*0.1:.2e}")


# ── Training epoch ───────────────────────────────────────────────────────────

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    max_grad_norm: float = 1.0,
) -> Dict:
    model.train()
    total_loss = cls_loss_sum = reg_loss_sum = 0.0
    all_preds  = []
    all_labels = []
    n_batches  = 0

    for batch in loader:
        frames  = batch["frames"].to(device)
        masks   = batch["masks"].to(device)
        times   = batch["times"].to(device)
        labels  = batch["label"].to(device)
        clin    = batch["clinical_feat"].to(device)

        optimizer.zero_grad()
        out = model(frames, masks, times, labels, clinical_feat=clin)
        out.total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss    += out.total_loss.item()
        cls_loss_sum  += out.cls_loss.item()
        reg_loss_sum  += out.reg_loss.item()

        preds = out.logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        n_batches += 1

    metrics = compute_metrics(all_preds, all_labels)
    return {
        "total_loss": total_loss  / n_batches,
        "cls_loss":   cls_loss_sum / n_batches,
        "reg_loss":   reg_loss_sum / n_batches,
        "accuracy":   metrics["accuracy"],
        "per_class":  metrics["per_class"],
    }


# ── Validation epoch (patient-level, myo-weighted) ───────────────────────────

@torch.no_grad()
def val_epoch(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """
    Patient-level evaluation with myocardium-pixel-weighted slice aggregation.
    Slices with more myocardium pixels contribute proportionally more to the
    patient-level logit (mid-cavity slices dominate, apical/basal contribute less).
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    # pid → list of (logit_tensor, myo_weight)
    patient_data:      Dict[str, list] = defaultdict(list)
    patient_label_map: Dict[str, int]  = {}

    for batch in loader:
        frames = batch["frames"].to(device)
        masks  = batch["masks"].to(device)
        times  = batch["times"].to(device)
        labels = batch["label"].to(device)
        clin   = batch["clinical_feat"].to(device)

        out        = model(frames, masks, times, labels, clinical_feat=clin)
        total_loss += out.total_loss.item()
        n_batches  += 1

        logits_cpu = out.logits.detach().cpu()

        for i, meta in enumerate(batch["meta"]):
            pid  = meta["patient_id"]
            w    = float(meta["n_myo_pixels"])
            patient_data[pid].append((logits_cpu[i], w))
            patient_label_map[pid] = labels[i].item()

    all_preds:  List[int] = []
    all_labels: List[int] = []

    for pid, data in patient_data.items():
        logit_stack = torch.stack([d[0] for d in data])          # (S, n_classes)
        weights     = torch.tensor([d[1] for d in data])         # (S,)
        weights     = weights / (weights.sum() + 1e-8)
        mean_logit  = (logit_stack * weights.unsqueeze(1)).sum(0) # (n_classes,)
        all_preds.append(int(mean_logit.argmax().item()))
        all_labels.append(patient_label_map[pid])

    metrics = compute_metrics(all_preds, all_labels)
    return {
        "total_loss": total_loss / max(n_batches, 1),
        "accuracy":   metrics["accuracy"],
        "per_class":  metrics["per_class"],
        "preds":      all_preds,
        "labels":     all_labels,
    }


# ── Stratified k-fold CV ─────────────────────────────────────────────────────

def stratified_kfold(
    march9_dir:   str,
    training_dir: str,
    patient_ids:  List[str],
    args,
    device:       torch.device,
    n_splits:     int = 5,
) -> Dict:

    labels_all = []
    for pid in patient_ids:
        info = parse_patient_info(training_dir, pid)
        labels_all.append(CLASS_TO_IDX.get(info["group"], 0))

    skf       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_preds, all_true = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, labels_all)):
        train_pids = [patient_ids[i] for i in train_idx]
        val_pids   = [patient_ids[i] for i in val_idx]

        _flush(f"\nFold {fold_idx+1}/{n_splits}: "
               f"{len(train_pids)} train / {len(val_pids)} val patients")

        # augment=True for training, False for validation
        train_ds = ACDCSliceDataset(march9_dir, training_dir, train_pids,
                                    augment=True)
        val_ds   = ACDCSliceDataset(march9_dir, training_dir, val_pids,
                                    augment=False)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=2, pin_memory=True)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=2, pin_memory=True)

        # Fresh model
        model = CardiacMotionODE(
            n_classes=5, ode_method='euler',
            n_clinical=N_CLINICAL_FEATURES,
        ).to(device)

        # Freeze registration net initially
        freeze_registration(model)
        _flush(f"  Registration net frozen for first {args.freeze_reg_epochs} epochs")

        # Optimizer — only non-frozen params at start
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6,
        )

        # W&B
        use_wandb = WANDB_AVAILABLE and getattr(args, "wandb_project", None)
        if use_wandb:
            run_name = f"{args.wandb_run_name or 'kfold'}_fold{fold_idx+1}"
            wandb.init(project=args.wandb_project, name=run_name,
                       config=vars(args), reinit=True,
                       tags=["kfold_v3", "full_model"])

        best_val_acc  = 0.0
        best_val_loss = float("inf")
        best_preds    = []
        best_labels   = []
        patience_ctr  = 0
        reg_unfrozen  = False

        for epoch in range(1, args.epochs + 1):
            epoch_t0 = time.time()

            # Unfreeze registration after freeze_reg_epochs
            if not reg_unfrozen and epoch > args.freeze_reg_epochs:
                unfreeze_registration(model, optimizer, args.lr)
                reg_unfrozen = True

            train_m = train_epoch(model, train_loader, optimizer, device)
            val_m   = val_epoch(model, val_loader, device)
            scheduler.step()
            elapsed = time.time() - epoch_t0

            # ── Per-epoch heartbeat (always printed) ─────────────────────────
            _flush(
                f"  [Fold {fold_idx+1} | Ep {epoch:3d}/{args.epochs}] "
                f"loss={train_m['total_loss']:.3f} "
                f"train_acc={train_m['accuracy']*100:.1f}% "
                f"val_acc={val_m['accuracy']*100:.1f}% "
                f"val_loss={val_m['total_loss']:.3f} "
                f"({elapsed:.0f}s)"
            )

            # W&B logging
            if use_wandb:
                wandb.log({
                    "epoch":          epoch,
                    "train/loss":     train_m["total_loss"],
                    "train/cls_loss": train_m["cls_loss"],
                    "train/reg_loss": train_m["reg_loss"],
                    "train/acc":      train_m["accuracy"],
                    "val/acc":        val_m["accuracy"],
                    "val/loss":       val_m["total_loss"],
                    "lr": scheduler.get_last_lr()[0],
                })

            # Track best predictions by val_acc (for confusion matrix)
            if val_m["accuracy"] > best_val_acc:
                best_val_acc  = val_m["accuracy"]
                best_preds    = val_m["preds"]
                best_labels   = val_m["labels"]
                if use_wandb:
                    wandb.summary["best_val_acc"] = best_val_acc

            # Early stopping on val_loss (more stable than val_acc)
            if val_m["total_loss"] < best_val_loss:
                best_val_loss = val_m["total_loss"]
                patience_ctr  = 0
            else:
                patience_ctr += 1

            if patience_ctr >= args.patience:
                _flush(f"  Early stopping at epoch {epoch} "
                       f"(val_loss hasn't improved for {args.patience} epochs)")
                break

        if use_wandb:
            wandb.finish()

        _flush(f"  Best val acc: {best_val_acc*100:.1f}%")
        all_preds.extend(best_preds)
        all_true.extend(best_labels)

    return _summarize_kfold(all_preds, all_true)


# ── LOPO CV ──────────────────────────────────────────────────────────────────

def lopo_cv(
    march9_dir:   str,
    training_dir: str,
    patient_ids:  List[str],
    args,
    device:       torch.device,
) -> Dict:
    patient_labels = {}
    for pid in patient_ids:
        info = parse_patient_info(training_dir, pid)
        patient_labels[pid] = CLASS_TO_IDX.get(info["group"], -1)

    _flush(f"\n{'='*60}")
    _flush(f"  Leave-One-Patient-Out CV  ({len(patient_ids)} folds)")
    _flush(f"{'='*60}\n")

    all_fold_results = []

    for fold_idx, held_out in enumerate(patient_ids):
        train_pids = [p for p in patient_ids if p != held_out]

        _flush(f"Fold {fold_idx+1:02d}/{len(patient_ids)} — held out: {held_out} "
               f"({IDX_TO_CLASS.get(patient_labels[held_out], '?')})")

        train_ds = ACDCSliceDataset(march9_dir, training_dir, train_pids,
                                    target_h=128, target_w=128, augment=True)
        test_ds  = ACDCSliceDataset(march9_dir, training_dir, [held_out],
                                    target_h=128, target_w=128, augment=False)

        if len(test_ds) == 0:
            _flush(f"  Skipping — no valid slices for {held_out}")
            continue

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=2, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                                  collate_fn=collate_fn, num_workers=2, pin_memory=True)

        model = CardiacMotionODE(n_classes=5, ode_method='euler',
                                 n_clinical=N_CLINICAL_FEATURES).to(device)
        freeze_registration(model)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6,
        )

        best_val_acc  = 0.0
        best_val_loss = float("inf")
        best_val_pred = None
        patience_ctr  = 0
        reg_unfrozen  = False

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            if not reg_unfrozen and epoch > args.freeze_reg_epochs:
                unfreeze_registration(model, optimizer, args.lr)
                reg_unfrozen = True

            train_m = train_epoch(model, train_loader, optimizer, device)
            val_m   = val_epoch(model, test_loader, device)
            scheduler.step()

            _flush(
                f"  [Fold {fold_idx+1:02d} | Ep {epoch:3d}/{args.epochs}] "
                f"train_acc={train_m['accuracy']*100:.1f}% "
                f"val_acc={val_m['accuracy']*100:.1f}% "
                f"val_loss={val_m['total_loss']:.3f} "
                f"({time.time()-t0:.0f}s)"
            )

            if val_m["accuracy"] > best_val_acc:
                best_val_acc  = val_m["accuracy"]
                best_val_pred = val_m["preds"]

            if val_m["total_loss"] < best_val_loss:
                best_val_loss = val_m["total_loss"]
                patience_ctr  = 0
            else:
                patience_ctr += 1

            if patience_ctr >= args.patience:
                _flush(f"  Early stopping at epoch {epoch}")
                break

        fold_result = {
            "fold":       fold_idx,
            "held_out":   held_out,
            "true_label": patient_labels[held_out],
            "pred_label": int(np.bincount(best_val_pred).argmax()) if best_val_pred else -1,
            "val_acc":    best_val_acc,
        }
        all_fold_results.append(fold_result)
        _flush(f"  Best val acc: {best_val_acc*100:.1f}%  "
               f"| Pred: {IDX_TO_CLASS.get(fold_result['pred_label'], '?')} "
               f"True: {IDX_TO_CLASS.get(patient_labels[held_out], '?')}\n")

    return _summarize_lopo(all_fold_results)


# ── Summarise results ─────────────────────────────────────────────────────────

def _summarize_kfold(all_preds, all_true) -> Dict:
    preds  = np.array(all_preds)
    labels = np.array(all_true)
    acc    = (preds == labels).mean()

    _flush(f"\n{'='*60}")
    _flush(f"  K-Fold CV Final Results")
    _flush(f"{'='*60}")
    _flush(f"  Overall accuracy: {acc*100:.1f}%  ({(preds==labels).sum()}/{len(labels)})")
    _flush(f"\n  Per-class accuracy:")
    for c in range(5):
        mask = labels == c
        if mask.sum() > 0:
            ca   = (preds[mask] == labels[mask]).mean()
            name = IDX_TO_CLASS.get(c, str(c))
            _flush(f"    {name:<6}: {ca*100:.1f}%  ({mask.sum()} patients)")

    _flush(f"\n  Confusion matrix (rows=true, cols=pred):")
    classes = [IDX_TO_CLASS[i] for i in range(5)]
    cm      = confusion_matrix(labels, preds, labels=list(range(5)))
    header  = "       " + "  ".join(f"{c:>5}" for c in classes)
    _flush(header)
    for i, row in enumerate(cm):
        _flush(f"  {classes[i]:<5}: " + "  ".join(f"{v:>5}" for v in row))
    _flush(f"{'='*60}\n")

    return {"accuracy": float(acc), "per_class": {}, "confusion_matrix": cm.tolist()}


def _summarize_lopo(results) -> Dict:
    preds  = np.array([r["pred_label"]  for r in results if r["pred_label"] != -1])
    labels = np.array([r["true_label"]  for r in results if r["pred_label"] != -1])
    acc    = (preds == labels).mean()

    _flush(f"\n{'='*60}")
    _flush(f"  LOPO CV Final Results")
    _flush(f"{'='*60}")
    _flush(f"  Overall accuracy: {acc*100:.1f}%  ({(preds==labels).sum()}/{len(labels)})")
    _flush(f"\n  Per-class accuracy:")
    for c in range(5):
        mask = labels == c
        if mask.sum() > 0:
            ca   = (preds[mask] == labels[mask]).mean()
            name = IDX_TO_CLASS.get(c, str(c))
            _flush(f"    {name:<6}: {ca*100:.1f}%  ({mask.sum()} patients)")

    _flush(f"\n  Confusion matrix (rows=true, cols=pred):")
    classes = [IDX_TO_CLASS[i] for i in range(5)]
    cm      = confusion_matrix(labels, preds, labels=list(range(5)))
    header  = "       " + "  ".join(f"{c:>5}" for c in classes)
    _flush(header)
    for i, row in enumerate(cm):
        _flush(f"  {classes[i]:<5}: " + "  ".join(f"{v:>5}" for v in row))
    _flush(f"{'='*60}\n")

    return {"accuracy": float(acc), "per_class": {}, "confusion_matrix": cm.tolist()}


# ── Argument parser ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CardiacMotionODE v3")
    p.add_argument("--march9_dir",        type=str, required=True)
    p.add_argument("--training_dir",      type=str, required=True)
    p.add_argument("--n_patients",        type=int, default=100)
    p.add_argument("--epochs",            type=int, default=100)
    p.add_argument("--batch_size",        type=int, default=8)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--patience",          type=int, default=15,
                   help="Early stopping patience on val_loss (default 15)")
    p.add_argument("--freeze_reg_epochs", type=int, default=20,
                   help="Freeze registration net for first N epochs (default 20)")
    p.add_argument("--log_every",         type=int, default=5,
                   help="(Unused — every epoch is printed; kept for compatibility)")
    p.add_argument("--cv",          type=str, default="kfold",
                   choices=["lopo", "kfold"])
    p.add_argument("--n_splits",    type=int, default=5)
    p.add_argument("--ablation",    type=str, default=None,
                   choices=[None, "no_ode", "no_graph"])
    p.add_argument("--checkpoint_dir",  type=str, default="checkpoints")
    p.add_argument("--results_file",    type=str, default="results_v3.json")
    p.add_argument("--wandb_project",   type=str, default=None)
    p.add_argument("--wandb_run_name",  type=str, default=None)
    return p.parse_args()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _flush(f"Device : {device}")
    _flush(f"Config : {vars(args)}\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    all_pids = discover_patients_from_images(args.march9_dir)[:args.n_patients]
    _flush(f"Using {len(all_pids)} patients: {all_pids[0]} … {all_pids[-1]}")

    from collections import defaultdict
    class_counts = defaultdict(int)
    for pid in all_pids:
        info = parse_patient_info(args.training_dir, pid)
        class_counts[info["group"]] += 1
    _flush(f"Class distribution: {dict(class_counts)}\n")

    _flush(f"Model variant: full_model (v3 — with clinical features)")
    _flush(f"Clinical features: {N_CLINICAL_FEATURES} scalars (EF, volumes, wall, RV/LV, H, W)")
    _flush(f"Regularisation: weight_decay=5e-4, label_smooth=0.1, GAT/ODE dropout=0.3/0.5")
    _flush(f"Augmentation: flip + rotate±15° + intensity jitter (train only)")
    _flush(f"Early stopping: patience={args.patience} epochs on val_loss")
    _flush(f"Reg freeze: first {args.freeze_reg_epochs} epochs\n")

    if args.cv == "lopo":
        results = lopo_cv(args.march9_dir, args.training_dir, all_pids, args, device)
    else:
        results = stratified_kfold(
            args.march9_dir, args.training_dir, all_pids, args, device, args.n_splits)

    results["config"]   = vars(args)
    results["ablation"] = "full_model_v3"

    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    _flush(f"Results saved to {args.results_file}")


if __name__ == "__main__":
    main()
