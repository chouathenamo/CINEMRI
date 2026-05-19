"""
ablation_base.py — Shared infrastructure for single-condition ablation scripts.
Imported by ablation_no_ode.py, ablation_no_graph.py, ablation_no_clinical.py.
Do not run directly.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add parent directory (src/) to path so model_v3, dataset_v3, etc. are importable
# Works whether scripts are run from src/ or src/ablation/
_this_dir   = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
sys.path.insert(0, _parent_dir)
sys.path.insert(0, _this_dir)

from dataset_v3 import (
    ACDCSliceDataset, collate_fn,
    IDX_TO_CLASS, CLASS_TO_IDX,
    discover_patients_from_images, parse_patient_info,
    N_CLINICAL_FEATURES,
)
from model_v3      import CardiacMotionODE
from ode_v3        import SequenceEncoder, ClassifierHead, geodesic_deviation
from graph_v3      import GraphMotionEncoder
from registration  import RegistrationNet, RegistrationLoss
from metrics_v3    import (compute_all_metrics, print_metrics_table,
                            metrics_to_latex, latex_table_header)


# ── Ablation model output ─────────────────────────────────────────────────────

@dataclass
class AblationOutput:
    logits:       torch.Tensor
    total_loss:   torch.Tensor
    cls_loss:     torch.Tensor
    reg_loss:     torch.Tensor
    bend_loss:    torch.Tensor
    fold_loss:    torch.Tensor
    z_traj:       Optional[torch.Tensor]
    phi_sequence: torch.Tensor
    geodesic_dev: Optional[torch.Tensor] = None


# ── no_ode model ──────────────────────────────────────────────────────────────

class CardiacMotionNoODE(nn.Module):
    """Registration → Graph → GRU → Classifier  (no ODE)."""

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
        self.classifier    = ClassifierHead(in_dim=d_z + n_clinical,
                                            n_classes=n_classes, dropout=0.5)

    def forward(self, frames, masks, times, labels=None, clinical_feat=None):
        B, N_frames, C, H, W = frames.shape
        N_pairs = N_frames - 1
        device  = frames.device
        all_embeddings, all_phis = [], []
        total_reg  = torch.tensor(0., device=device)
        total_bend = torch.tensor(0., device=device)
        total_fold = torch.tensor(0., device=device)
        for t in range(N_pairs):
            warped, vel, phi = self.registration.get_warped(frames[:, t], frames[:, t+1])
            all_phis.append(phi)
            _, ncc, bend, fold = self.reg_loss_fn(warped, frames[:, t], vel, phi, masks[:, t])
            total_reg  += ncc;  total_bend += bend;  total_fold += fold
            all_embeddings.append(self.graph_encoder(masks[:, t], phi))
        total_reg  /= N_pairs;  total_bend /= N_pairs;  total_fold /= N_pairs
        emb_seq = torch.stack(all_embeddings, dim=1)
        phi_seq = torch.stack(all_phis,       dim=1)
        z0      = self.seq_encoder(emb_seq)
        cf = clinical_feat.to(z0.device) if (clinical_feat is not None and self.n_clinical > 0) \
             else torch.zeros(z0.shape[0], self.n_clinical, device=z0.device)
        logits  = self.classifier(torch.cat([z0, cf], dim=1))
        cls_loss = _cls_loss(logits, labels, device) if labels is not None \
                   else torch.tensor(0., device=device)
        total    = cls_loss + self.alpha * total_reg + self.beta * (total_bend + total_fold)
        return AblationOutput(logits=logits, total_loss=total, cls_loss=cls_loss,
                              reg_loss=total_reg, bend_loss=total_bend,
                              fold_loss=total_fold, z_traj=None, phi_sequence=phi_seq)


# ── no_graph model ────────────────────────────────────────────────────────────

class PhiCNNEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2,  16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_dim, 3, stride=2, padding=1), nn.ReLU(),
        )
    def forward(self, masks, phis):
        return self.net(phis).mean(dim=[2, 3])


class CardiacMotionNoGraph(nn.Module):
    """Registration → CNN(phi) → GRU → ODE → Classifier  (no GAT)."""

    def __init__(self, n_classes=5, d_z=64, alpha=0.1, beta=0.01,
                 ode_method='euler', n_clinical=8):
        super().__init__()
        self.alpha       = alpha
        self.beta        = beta
        self.n_clinical  = n_clinical
        self.registration  = RegistrationNet()
        self.reg_loss_fn   = RegistrationLoss(alpha=alpha, beta=beta)
        self.phi_encoder   = PhiCNNEncoder(out_dim=d_z)
        from ode_v3 import CardiacODEClassifier
        self.ode_classifier = CardiacODEClassifier(
            d_z=d_z, n_classes=n_classes, method=ode_method, n_clinical=n_clinical)

    def forward(self, frames, masks, times, labels=None, clinical_feat=None):
        B, N_frames, C, H, W = frames.shape
        N_pairs = N_frames - 1
        device  = frames.device
        all_embeddings, all_phis = [], []
        total_reg  = torch.tensor(0., device=device)
        total_bend = torch.tensor(0., device=device)
        total_fold = torch.tensor(0., device=device)
        for t in range(N_pairs):
            warped, vel, phi = self.registration.get_warped(frames[:, t], frames[:, t+1])
            all_phis.append(phi)
            _, ncc, bend, fold = self.reg_loss_fn(warped, frames[:, t], vel, phi, masks[:, t])
            total_reg  += ncc;  total_bend += bend;  total_fold += fold
            all_embeddings.append(self.phi_encoder(masks[:, t], phi))
        total_reg  /= N_pairs;  total_bend /= N_pairs;  total_fold /= N_pairs
        emb_seq   = torch.stack(all_embeddings, dim=1)
        phi_seq   = torch.stack(all_phis,       dim=1)
        times_mid = (times[:, :-1] + times[:, 1:]) / 2.0
        logits, z_traj = self.ode_classifier(emb_seq, times_mid, clinical_feat=clinical_feat)
        cls_loss = _cls_loss(logits, labels, device) if labels is not None \
                   else torch.tensor(0., device=device)
        total    = cls_loss + self.alpha * total_reg + self.beta * (total_bend + total_fold)
        with torch.no_grad():
            geo_dev = geodesic_deviation(z_traj)
        return AblationOutput(logits=logits, total_loss=total, cls_loss=cls_loss,
                              reg_loss=total_reg, bend_loss=total_bend,
                              fold_loss=total_fold, z_traj=z_traj,
                              phi_sequence=phi_seq, geodesic_dev=geo_dev)


# ── Shared loss helper ────────────────────────────────────────────────────────

def _cls_loss(logits, labels, device):
    w = torch.ones(5, device=device)
    w[CLASS_TO_IDX["MINF"]] = 2.0
    w[CLASS_TO_IDX["DCM"]]  = 1.5
    return F.cross_entropy(logits, labels, weight=w, label_smoothing=0.1)


# ── Training helpers ──────────────────────────────────────────────────────────

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
    optimizer.add_param_group(
        {"params": [p for p in model.registration.parameters() if p.requires_grad],
         "lr": base_lr * 0.1})
    _flush(f"    [Unfreeze] Registration → optimizer at LR={base_lr*0.1:.2e}")


def train_epoch(model, loader, optimizer, device, max_grad_norm=1.0):
    model.train()
    tot_loss = cls_sum = reg_sum = 0.0
    all_preds, all_labels = [], []
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
        tot_loss += out.total_loss.item()
        cls_sum  += out.cls_loss.item()
        reg_sum  += out.reg_loss.item()
        all_preds.extend(out.logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    n = max(len(loader), 1)
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return {"total_loss": tot_loss/n, "cls_loss": cls_sum/n,
            "reg_loss": reg_sum/n, "accuracy": acc}


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    tot_loss = 0.0
    patient_data: Dict[str, list] = defaultdict(list)
    patient_labels: Dict[str, int] = {}
    for batch in loader:
        frames = batch["frames"].to(device)
        masks  = batch["masks"].to(device)
        times  = batch["times"].to(device)
        labels = batch["label"].to(device)
        clin   = batch["clinical_feat"].to(device)
        out = model(frames, masks, times, labels, clinical_feat=clin)
        tot_loss += out.total_loss.item()
        for i, meta in enumerate(batch["meta"]):
            pid = meta["patient_id"]
            patient_data[pid].append((out.logits[i].cpu(), float(meta["n_myo_pixels"])))
            patient_labels[pid] = labels[i].item()
    all_preds, all_true = [], []
    for pid, data in patient_data.items():
        logits  = torch.stack([d[0] for d in data])
        weights = torch.tensor([d[1] for d in data])
        weights = weights / (weights.sum() + 1e-8)
        all_preds.append(int((logits * weights.unsqueeze(1)).sum(0).argmax()))
        all_true.append(patient_labels[pid])
    acc = (np.array(all_preds) == np.array(all_true)).mean()
    return {"total_loss": tot_loss / max(len(loader), 1),
            "accuracy": acc, "preds": all_preds, "labels": all_true}


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_fold_checkpoint(condition, fold_idx, all_preds, all_true,
                         best_val_acc, epoch_stopped):
    """Save after each completed fold so crashes never lose work."""
    path = f"ckpt_{condition}_fold{fold_idx+1}.json"
    with open(path, "w") as f:
        json.dump({
            "condition":     condition,
            "fold":          fold_idx + 1,
            "folds_done":    fold_idx + 1,
            "all_preds":     all_preds,
            "all_true":      all_true,
            "best_val_acc":  best_val_acc,
            "epoch_stopped": epoch_stopped,
        }, f, indent=2)
    _flush(f"  ✓ Fold checkpoint saved → {path}")
    return path


def load_existing_checkpoints(condition, n_splits):
    """Load all completed fold checkpoints for a condition."""
    all_preds, all_true = [], []
    start_fold = 0
    for fold_idx in range(n_splits):
        path = f"ckpt_{condition}_fold{fold_idx+1}.json"
        if os.path.exists(path):
            with open(path) as f:
                ckpt = json.load(f)
            all_preds.extend(ckpt["all_preds"])
            all_true.extend(ckpt["all_true"])
            start_fold = fold_idx + 1
            _flush(f"  ✓ Loaded fold {fold_idx+1} checkpoint  "
                   f"(best_val_acc={ckpt['best_val_acc']*100:.1f}%  "
                   f"stopped ep={ckpt['epoch_stopped']})")
        else:
            break
    return all_preds, all_true, start_fold


# ── Main training loop for one condition ─────────────────────────────────────

def run_ablation(condition: str, model_factory, args, device):
    """
    Full k-fold CV for one ablation condition.
    model_factory: callable() → nn.Module
    """
    # ── Write PID file so you always know which process to keep ──────────────
    pid_file = f"pid_{condition}.txt"
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
    _flush(f"\n{'━'*64}")
    _flush(f"  CONDITION : {condition.upper()}")
    _flush(f"  PID       : {os.getpid()}  (written to {pid_file})")
    _flush(f"{'━'*64}")

    # ── Patient list and fold splits ──────────────────────────────────────────
    all_pids = discover_patients_from_images(args.march9_dir)[:args.n_patients]
    labels_all = []
    class_counts = defaultdict(int)
    for pid in all_pids:
        info = parse_patient_info(args.training_dir, pid)
        idx  = CLASS_TO_IDX.get(info["group"], 0)
        labels_all.append(idx)
        class_counts[info["group"]] += 1
    _flush(f"  Patients  : {len(all_pids)}  ({all_pids[0]} … {all_pids[-1]})")
    _flush(f"  Classes   : {dict(class_counts)}")

    skf         = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    fold_splits = list(skf.split(all_pids, labels_all))
    _flush(f"  Fold sizes: " +
           "  ".join(f"fold{i+1}:{len(v)}val" for i,(_, v) in enumerate(fold_splits)))

    # ── Resume from existing checkpoints ─────────────────────────────────────
    all_preds, all_true, start_fold = load_existing_checkpoints(condition, args.n_splits)
    if start_fold > 0:
        _flush(f"\n  Resuming from fold {start_fold+1} "
               f"({start_fold} folds already done, {len(all_preds)} patients loaded)")

    # ── K-fold loop ───────────────────────────────────────────────────────────
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        if fold_idx < start_fold:
            _flush(f"\n  Fold {fold_idx+1}/{args.n_splits}: SKIPPED (checkpoint exists)")
            continue

        train_pids = [all_pids[i] for i in train_idx]
        val_pids   = [all_pids[i] for i in val_idx]
        _flush(f"\n{'─'*64}")
        _flush(f"  Fold {fold_idx+1}/{args.n_splits}  "
               f"({len(train_pids)} train / {len(val_pids)} val)")
        _flush(f"{'─'*64}")

        train_ds = ACDCSliceDataset(args.march9_dir, args.training_dir,
                                    train_pids, augment=True)
        val_ds   = ACDCSliceDataset(args.march9_dir, args.training_dir,
                                    val_pids,   augment=False)
        _flush(f"  Train slices: {len(train_ds)}   Val slices: {len(val_ds)}")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True,  collate_fn=collate_fn,
                                  num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate_fn,
                                  num_workers=0, pin_memory=True)

        model     = model_factory().to(device)
        freeze_registration(model)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)

        # W&B
        use_wandb = WANDB_AVAILABLE and getattr(args, "wandb_project", None)
        if use_wandb:
            run_name = f"abl_{condition}_fold{fold_idx+1}"
            wandb.init(project=args.wandb_project, name=run_name,
                       config={**vars(args), "condition": condition},
                       reinit=True, tags=["ablation_v3", condition])

        best_val_acc  = 0.0
        best_val_loss = float("inf")
        best_preds    = []
        best_labels   = []
        patience_ctr  = 0
        reg_unfrozen  = False
        epoch_stopped = args.epochs

        _flush(f"\n  {'Ep':>4}  {'TrainLoss':>10}  {'ClsLoss':>8}  "
               f"{'RegLoss':>8}  {'TrainAcc':>9}  {'ValAcc':>7}  "
               f"{'ValLoss':>8}  {'Pat':>4}  {'Time':>6}")
        _flush(f"  {'─'*4}  {'─'*10}  {'─'*8}  {'─'*8}  "
               f"{'─'*9}  {'─'*7}  {'─'*8}  {'─'*4}  {'─'*6}")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            if not reg_unfrozen and epoch > args.freeze_reg_epochs:
                unfreeze_registration(model, optimizer, args.lr)
                reg_unfrozen = True
                _flush(f"  [Epoch {epoch}] Registration unfrozen at LR={args.lr*0.1:.2e}")

            train_m = train_epoch(model, train_loader, optimizer, device)
            val_m   = val_epoch(model, val_loader, device)
            scheduler.step()
            elapsed = time.time() - t0

            # ── Verbose per-epoch row ─────────────────────────────────────────
            _flush(
                f"  {epoch:>4}  "
                f"{train_m['total_loss']:>10.4f}  "
                f"{train_m['cls_loss']:>8.4f}  "
                f"{train_m['reg_loss']:>8.4f}  "
                f"{train_m['accuracy']*100:>8.1f}%  "
                f"{val_m['accuracy']*100:>6.1f}%  "
                f"{val_m['total_loss']:>8.4f}  "
                f"{patience_ctr:>4}  "
                f"{elapsed:>5.0f}s"
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
                    "lr":             scheduler.get_last_lr()[0],
                    "patience":       patience_ctr,
                })

            if val_m["accuracy"] > best_val_acc:
                best_val_acc  = val_m["accuracy"]
                best_preds    = val_m["preds"]
                best_labels   = val_m["labels"]
                _flush(f"  ↑ New best val_acc: {best_val_acc*100:.1f}%")

            if val_m["total_loss"] < best_val_loss:
                best_val_loss = val_m["total_loss"]
                patience_ctr  = 0
            else:
                patience_ctr += 1

            if patience_ctr >= args.patience:
                epoch_stopped = epoch
                _flush(f"  Early stopping at epoch {epoch} "
                       f"(patience {args.patience} exceeded)")
                break

        if use_wandb:
            wandb.finish()

        _flush(f"\n  Fold {fold_idx+1} done.  Best val_acc={best_val_acc*100:.1f}%  "
               f"Stopped at epoch={epoch_stopped}")

        # ── Per-fold val confusion matrix ─────────────────────────────────────
        fold_metrics = compute_all_metrics(best_preds, best_labels)
        _flush(f"  Fold {fold_idx+1} val confusion matrix:")
        cm = fold_metrics["confusion_matrix"]
        classes = [IDX_TO_CLASS[i] for i in range(5)]
        _flush("    " + "  ".join(f"{c:>5}" for c in classes))
        for i, row in enumerate(cm):
            _flush(f"  {classes[i]}: " + "  ".join(f"{v:>5}" for v in row))

        all_preds.extend(best_preds)
        all_true.extend(best_labels)

        # ── Save fold checkpoint ──────────────────────────────────────────────
        save_fold_checkpoint(condition, fold_idx, all_preds, all_true,
                             best_val_acc, epoch_stopped)

    # ── Final aggregate metrics ───────────────────────────────────────────────
    metrics = compute_all_metrics(all_preds, all_true)
    print_metrics_table(metrics, title=f"FINAL — {condition}")

    result = {
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
    with open(args.results_file, "w") as f:
        json.dump({**result, "config": vars(args)}, f, indent=2)
    _flush(f"\n  Results saved → {args.results_file}")
    _flush(f"\n  LaTeX row:")
    _flush("  " + latex_table_header())
    _flush("  " + metrics_to_latex(metrics, model_name=condition.replace("_", "\\_")))
    _flush("  \\bottomrule\n  \\end{tabular}")

    # Clean up PID file
    if os.path.exists(pid_file):
        os.remove(pid_file)

    return result


# ── Shared argparser ──────────────────────────────────────────────────────────

def make_parser(condition_name: str):
    import argparse
    p = argparse.ArgumentParser(
        description=f"Ablation: {condition_name} on CardiacMotionODE v3")
    p.add_argument("--march9_dir",        type=str, required=True)
    p.add_argument("--training_dir",      type=str, required=True)
    p.add_argument("--n_patients",        type=int, default=100)
    p.add_argument("--epochs",            type=int, default=120)
    p.add_argument("--batch_size",        type=int, default=8)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--patience",          type=int, default=30)
    p.add_argument("--freeze_reg_epochs", type=int, default=20)
    p.add_argument("--n_splits",          type=int, default=5)
    p.add_argument("--wandb_project",     type=str, default=None)
    p.add_argument("--results_file",      type=str,
                   default=f"results_abl_{condition_name}.json")
    p.add_argument("--resume",            action="store_true",
                   help="Auto-resume from existing fold checkpoints")
    return p
