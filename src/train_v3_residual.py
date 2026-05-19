"""
train_v3_residual.py — identical to train_v3.py with one change:
    from model_v3_residual import ...   (was: from model_v3 import ...)

Run:
    mkdir -p logs && nohup python train_v3_residual.py \\
        --march9_dir  /home/amo/CINEMRI/data/ACDC/March9Data \\
        --training_dir /home/amo/CINEMRI/data/ACDC/training \\
        --n_patients 100 --epochs 120 --batch_size 8 \\
        --n_splits 5 --patience 30 --log_every 5 --cv kfold \\
        --results_file results_residual_p30.json \\
        --wandb_project cardiac-ode --wandb_run_name residual_p30 \\
        > logs/residual_p30.txt 2>&1 &
    echo "PID: $!"
    tail -f logs/residual_p30.txt
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
from model_v3_residual import CardiacMotionODE, ModelOutput          # ← only change


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
    for p in model.registration.parameters():
        p.requires_grad = True
    reg_params = [p for p in model.registration.parameters() if p.requires_grad]
    optimizer.add_param_group({"params": reg_params, "lr": base_lr * 0.1})
    _flush(f"  [Unfreeze] Registration net added to optimizer at LR={base_lr*0.1:.2e}")


# ── Training epoch ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, max_grad_norm=1.0):
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


# ── Validation epoch ─────────────────────────────────────────────────────────

@torch.no_grad()
def val_epoch(model, loader, device):
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
        logit_stack = torch.stack([d[0] for d in data])
        weights     = torch.tensor([d[1] for d in data])
        weights     = weights / (weights.sum() + 1e-8)
        mean_logit  = (logit_stack * weights.unsqueeze(1)).sum(0)
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

def stratified_kfold(march9_dir, training_dir, patient_ids, args, device, n_splits=5):
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

        train_ds = ACDCSliceDataset(march9_dir, training_dir, train_pids, augment=True)
        val_ds   = ACDCSliceDataset(march9_dir, training_dir, val_pids,   augment=False)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                                  collate_fn=collate_fn, num_workers=2, pin_memory=True)

        model = CardiacMotionODE(
            n_classes=5, ode_method='euler', n_clinical=N_CLINICAL_FEATURES,
        ).to(device)

        freeze_registration(model)
        _flush(f"  Registration net frozen for first {args.freeze_reg_epochs} epochs")

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6,
        )

        use_wandb = WANDB_AVAILABLE and getattr(args, "wandb_project", None)
        if use_wandb:
            run_name = f"{args.wandb_run_name or 'kfold'}_fold{fold_idx+1}"
            wandb.init(project=args.wandb_project, name=run_name,
                       config=vars(args), reinit=True,
                       tags=["kfold_v3", "residual"])

        best_val_acc  = 0.0
        best_val_loss = float("inf")
        best_preds    = []
        best_labels   = []
        patience_ctr  = 0
        reg_unfrozen  = False

        for epoch in range(1, args.epochs + 1):
            epoch_t0 = time.time()

            if not reg_unfrozen and epoch > args.freeze_reg_epochs:
                unfreeze_registration(model, optimizer, args.lr)
                reg_unfrozen = True

            train_m = train_epoch(model, train_loader, optimizer, device)
            val_m   = val_epoch(model, val_loader, device)
            scheduler.step()
            elapsed = time.time() - epoch_t0

            _flush(
                f"  [Fold {fold_idx+1} | Ep {epoch:3d}/{args.epochs}] "
                f"loss={train_m['total_loss']:.3f} "
                f"train_acc={train_m['accuracy']*100:.1f}% "
                f"val_acc={val_m['accuracy']*100:.1f}% "
                f"val_loss={val_m['total_loss']:.3f} "
                f"({elapsed:.0f}s)"
            )

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

            if val_m["accuracy"] > best_val_acc:
                best_val_acc  = val_m["accuracy"]
                best_preds    = val_m["preds"]
                best_labels   = val_m["labels"]
                if use_wandb:
                    wandb.summary["best_val_acc"] = best_val_acc

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


# ── Summarise results ─────────────────────────────────────────────────────────

def _summarize_kfold(all_preds, all_true):
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


# ── Argument parser ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train CardiacMotionODE (residual variant)")
    p.add_argument("--march9_dir",        type=str, required=True)
    p.add_argument("--training_dir",      type=str, required=True)
    p.add_argument("--n_patients",        type=int, default=100)
    p.add_argument("--epochs",            type=int, default=100)
    p.add_argument("--batch_size",        type=int, default=8)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--patience",          type=int, default=30)
    p.add_argument("--freeze_reg_epochs", type=int, default=20)
    p.add_argument("--log_every",         type=int, default=5)
    p.add_argument("--cv",                type=str, default="kfold",
                   choices=["lopo", "kfold"])
    p.add_argument("--n_splits",          type=int, default=5)
    p.add_argument("--ablation",          type=str, default=None)
    p.add_argument("--checkpoint_dir",    type=str, default="checkpoints")
    p.add_argument("--results_file",      type=str, default="results_residual.json")
    p.add_argument("--wandb_project",     type=str, default=None)
    p.add_argument("--wandb_run_name",    type=str, default=None)
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

    _flush(f"Model variant: residual ODE (z_final = z0 + z_traj[-1])")

    results = stratified_kfold(
        args.march9_dir, args.training_dir, all_pids, args, device, args.n_splits)

    results["config"]   = vars(args)
    results["ablation"] = "residual_ode"

    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    _flush(f"Results saved to {args.results_file}")


if __name__ == "__main__":
    main()
