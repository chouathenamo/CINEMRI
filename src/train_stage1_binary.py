"""
train_stage1_binary.py — Stage 1: Binary NOR vs DISEASE classifier
===================================================================
Uses the no-ODE architecture (GraphMotionEncoder + GRU + classifier)
with n_classes=2 (NOR=0, DISEASE=1).

NOR vs disease is a much simpler problem than 5-class — EF and
cardiac volumes alone strongly separate healthy from diseased hearts.
A dedicated binary classifier should achieve much higher NOR recall
than any 5-class model.

Saves per-patient predictions to:
    results/stage1_preds.json  (used by evaluate_pipeline.py)

Run:
    mkdir -p results logs && nohup python train_stage1_binary.py \\
        --march9_dir  /home/amo/CINEMRI/data/ACDC/March9Data \\
        --training_dir /home/amo/CINEMRI/data/ACDC/training \\
        --n_patients 100 --epochs 120 --batch_size 8 \\
        --n_splits 5 --patience 30 --seed 42 \\
        --results_file results/stage1_binary.json \\
        --wandb_project cardiac-ode --wandb_run_name stage1_binary \\
        > logs/stage1_binary.txt 2>&1 &
    echo "PID: $!"
    tail -f logs/stage1_binary.txt
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
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

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
from registration import RegistrationNet, RegistrationLoss
from graph_v3    import GraphMotionEncoder


# ── Binary label mapping ──────────────────────────────────────────────────────
# 0 = NOR, 1 = DISEASE (DCM / HCM / MINF / RV)
BINARY_NAMES = {0: "NOR", 1: "DISEASE"}

def to_binary(label: int) -> int:
    """Convert 5-class label to binary: NOR=0, everything else=1."""
    return 0 if label == CLASS_TO_IDX["NOR"] else 1


# ── Lightweight binary classifier head ───────────────────────────────────────

class BinaryClassifierHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),          # 2 classes: NOR / DISEASE
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ── Binary model output ───────────────────────────────────────────────────────

@dataclass
class BinaryModelOutput:
    logits:     torch.Tensor
    total_loss: torch.Tensor
    cls_loss:   torch.Tensor
    reg_loss:   torch.Tensor
    bend_loss:  torch.Tensor
    fold_loss:  torch.Tensor


# ── Stage 1 model: no-ODE, binary output ─────────────────────────────────────

class Stage1BinaryModel(nn.Module):
    """
    GraphMotionEncoder → GRU → BinaryClassifierHead
    No ODE — just the motion graph + GRU, proven to be the best
    architecture for this dataset.
    Binary output: NOR (0) vs DISEASE (1).
    """

    def __init__(
        self,
        d_z:         int   = 64,
        n_verts:     int   = 64,
        k_neighbors: int   = 6,
        alpha:       float = 0.1,
        beta:        float = 0.01,
        n_clinical:  int   = 8,
    ):
        super().__init__()
        self.alpha     = alpha
        self.beta      = beta
        self.n_clinical = n_clinical

        self.registration  = RegistrationNet()
        self.reg_loss_fn   = RegistrationLoss(alpha=alpha, beta=beta)
        self.graph_encoder = GraphMotionEncoder(
            n_verts=n_verts, k=k_neighbors, out_dim=d_z)
        self.gru           = nn.GRU(input_size=d_z, hidden_size=d_z,
                                    num_layers=1, batch_first=True)
        self.classifier    = BinaryClassifierHead(
            in_dim=d_z + n_clinical, dropout=0.5)

    def forward(
        self,
        frames:        torch.Tensor,              # (B, N_frames, 1, H, W)
        masks:         torch.Tensor,
        times:         torch.Tensor,
        labels:        Optional[torch.Tensor] = None,   # binary labels (B,)
        clinical_feat: Optional[torch.Tensor] = None,
    ) -> BinaryModelOutput:

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
        _, h_n         = self.gru(embeddings_seq)
        z0             = h_n.squeeze(0)                       # (B, d_z)

        if clinical_feat is not None and self.n_clinical > 0:
            cf   = clinical_feat.to(z0.device)
            z_in = torch.cat([z0, cf], dim=1)
        else:
            pad  = torch.zeros(z0.shape[0], self.n_clinical, device=z0.device)
            z_in = torch.cat([z0, pad], dim=1)

        logits = self.classifier(z_in)   # (B, 2)

        if labels is not None:
            # Up-weight NOR slightly since it's harder to recall
            weight = torch.tensor([4.5, 1.0], device=device)
            cls_loss = F.cross_entropy(logits, labels, weight=weight,
                                       label_smoothing=0.05)
        else:
            cls_loss = torch.tensor(0.0, device=device)

        total_loss = (cls_loss
                      + self.alpha * total_reg_loss
                      + self.beta  * (total_bend_loss + total_fold_loss))

        return BinaryModelOutput(
            logits     = logits,
            total_loss = total_loss,
            cls_loss   = cls_loss,
            reg_loss   = total_reg_loss,
            bend_loss  = total_bend_loss,
            fold_loss  = total_fold_loss,
        )


# ── Training helpers ──────────────────────────────────────────────────────────

def _flush(*args):
    print(*args, flush=True)


def freeze_registration(model):
    for p in model.registration.parameters():
        p.requires_grad = False


def unfreeze_registration(model, optimizer, base_lr):
    for p in model.registration.parameters():
        p.requires_grad = True
    optimizer.add_param_group({
        "params": [p for p in model.registration.parameters()
                   if p.requires_grad],
        "lr": base_lr * 0.1,
    })
    _flush(f"  [Unfreeze] Registration at LR={base_lr*0.1:.2e}")


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    n = 0

    for batch in loader:
        frames = batch["frames"].to(device)
        masks  = batch["masks"].to(device)
        times  = batch["times"].to(device)
        clin   = batch["clinical_feat"].to(device)
        # Convert to binary labels
        labels_5 = batch["label"].to(device)
        labels_b = torch.tensor([to_binary(l.item()) for l in labels_5],
                                 device=device)

        optimizer.zero_grad()
        out = model(frames, masks, times, labels_b, clinical_feat=clin)
        out.total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += out.total_loss.item()
        preds = out.logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels_b.cpu().tolist())
        n += 1

    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return {"total_loss": total_loss / n, "accuracy": acc}


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    patient_data:      Dict[str, list] = defaultdict(list)
    patient_label_map: Dict[str, int]  = {}
    patient_orig_label: Dict[str, int] = {}   # original 5-class label

    for batch in loader:
        frames   = batch["frames"].to(device)
        masks    = batch["masks"].to(device)
        times    = batch["times"].to(device)
        clin     = batch["clinical_feat"].to(device)
        labels_5 = batch["label"].to(device)
        labels_b = torch.tensor([to_binary(l.item()) for l in labels_5],
                                 device=device)

        out        = model(frames, masks, times, labels_b, clinical_feat=clin)
        total_loss += out.total_loss.item()
        n          += 1

        logits_cpu = out.logits.detach().cpu()
        for i, meta in enumerate(batch["meta"]):
            pid = meta["patient_id"]
            w   = float(meta["n_myo_pixels"])
            patient_data[pid].append((logits_cpu[i], w))
            patient_label_map[pid]   = labels_b[i].item()
            patient_orig_label[pid]  = labels_5[i].item()

    all_preds_b, all_labels_b = [], []
    all_pids, all_orig_labels = [], []

    for pid, data in patient_data.items():
        logit_stack = torch.stack([d[0] for d in data])
        weights     = torch.tensor([d[1] for d in data])
        weights     = weights / (weights.sum() + 1e-8)
        mean_logit  = (logit_stack * weights.unsqueeze(1)).sum(0)
        pred_b      = int(mean_logit.argmax().item())

        all_preds_b.append(pred_b)
        all_labels_b.append(patient_label_map[pid])
        all_pids.append(pid)
        all_orig_labels.append(patient_orig_label[pid])

    preds_b  = np.array(all_preds_b)
    labels_b = np.array(all_labels_b)
    acc      = (preds_b == labels_b).mean()

    # Recall for NOR and DISEASE separately
    nor_mask = labels_b == 0
    nor_recall = (preds_b[nor_mask] == 0).mean() if nor_mask.sum() > 0 else 0.0
    dis_recall = (preds_b[~nor_mask] == 1).mean() if (~nor_mask).sum() > 0 else 0.0

    return {
        "total_loss":  total_loss / max(n, 1),
        "accuracy":    float(acc),
        "nor_recall":  float(nor_recall),
        "dis_recall":  float(dis_recall),
        "preds_b":     all_preds_b,
        "labels_b":    all_labels_b,
        "pids":        all_pids,
        "orig_labels": all_orig_labels,
    }


# ── K-fold CV ─────────────────────────────────────────────────────────────────

def stratified_kfold(march9_dir, training_dir, patient_ids, args, device):
    # Use original 5-class labels for stratification (ensures balanced folds)
    labels_all = []
    for pid in patient_ids:
        info = parse_patient_info(training_dir, pid)
        labels_all.append(CLASS_TO_IDX.get(info["group"], 0))

    skf = StratifiedKFold(n_splits=args.n_splits,
                          shuffle=True, random_state=args.seed)

    all_preds_b   = []
    all_labels_b  = []
    all_pids      = []
    all_orig      = []   # original 5-class labels

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(patient_ids, labels_all)):

        train_pids = [patient_ids[i] for i in train_idx]
        val_pids   = [patient_ids[i] for i in val_idx]
        _flush(f"\nFold {fold_idx+1}/{args.n_splits}: "
               f"{len(train_pids)} train / {len(val_pids)} val  "
               f"[seed={args.seed}]")

        train_ds = ACDCSliceDataset(march9_dir, training_dir,
                                    train_pids, augment=True)
        val_ds   = ACDCSliceDataset(march9_dir, training_dir,
                                    val_pids,   augment=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_fn,
                                  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate_fn,
                                  num_workers=2, pin_memory=True)

        model = Stage1BinaryModel(n_clinical=N_CLINICAL_FEATURES).to(device)
        freeze_registration(model)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6,
        )

        use_wandb = WANDB_AVAILABLE and getattr(args, "wandb_project", None)
        if use_wandb:
            wandb.init(project=args.wandb_project,
                       name=f"{args.wandb_run_name}_fold{fold_idx+1}",
                       config=vars(args), reinit=True,
                       tags=["stage1", "binary"])

        best_val_acc   = 0.0
        best_balanced  = 0.0
        best_val_loss  = float("inf")
        best_val_data  = {}
        patience_ctr   = 0
        reg_unfrozen   = False

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            if not reg_unfrozen and epoch > args.freeze_reg_epochs:
                unfreeze_registration(model, optimizer, args.lr)
                reg_unfrozen = True

            train_m = train_epoch(model, train_loader, optimizer, device)
            val_m   = val_epoch(model, val_loader, device)
            scheduler.step()

            _flush(
                f"  [Fold {fold_idx+1} | Ep {epoch:3d}/{args.epochs}] "
                f"loss={train_m['total_loss']:.3f} "
                f"train_acc={train_m['accuracy']*100:.1f}% "
                f"val_acc={val_m['accuracy']*100:.1f}% "
                f"NOR_recall={val_m['nor_recall']*100:.1f}% "
                f"DIS_recall={val_m['dis_recall']*100:.1f}% "
                f"val_loss={val_m['total_loss']:.3f} "
                f"pat={patience_ctr}/{args.patience} "
                f"({time.time()-t0:.0f}s)"
            )

            if use_wandb:
                wandb.log({
                    "epoch":       epoch,
                    "train/loss":  train_m["total_loss"],
                    "train/acc":   train_m["accuracy"],
                    "val/acc":     val_m["accuracy"],
                    "val/loss":    val_m["total_loss"],
                    "val/nor_recall": val_m["nor_recall"],
                    "val/dis_recall": val_m["dis_recall"],
                    "patience":    patience_ctr,
                    "lr":          scheduler.get_last_lr()[0],
                })

            # if val_m["accuracy"] > best_val_acc:
            #     best_val_acc  = val_m["accuracy"]
            #     best_val_data = val_m
            balanced = (val_m["nor_recall"] + val_m["dis_recall"]) / 2
            if balanced > best_balanced:
                best_balanced = balanced
                best_val_data = val_m

            if val_m["total_loss"] < best_val_loss:
                best_val_loss = val_m["total_loss"]
                patience_ctr  = 0
            else:
                patience_ctr += 1

            if patience_ctr >= args.patience:
                _flush(f"  Early stopping at epoch {epoch}")
                break

        if use_wandb:
            wandb.finish()

        _flush(f"  Best val acc: {best_val_acc*100:.1f}%  "
               f"NOR recall: {best_val_data.get('nor_recall',0)*100:.1f}%  "
               f"DISEASE recall: {best_val_data.get('dis_recall',0)*100:.1f}%")

        all_preds_b.extend(best_val_data["preds_b"])
        all_labels_b.extend(best_val_data["labels_b"])
        all_pids.extend(best_val_data["pids"])
        all_orig.extend(best_val_data["orig_labels"])

    return _summarize(all_preds_b, all_labels_b,
                      all_pids, all_orig, args.results_file)


def _summarize(preds_b, labels_b, pids, orig_labels, results_file):
    preds_b  = np.array(preds_b)
    labels_b = np.array(labels_b)
    acc      = (preds_b == labels_b).mean()
    cm       = confusion_matrix(labels_b, preds_b, labels=[0, 1])

    nor_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    dis_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0
    precision_nor = cm[0, 0] / cm[:, 0].sum() if cm[:, 0].sum() > 0 else 0
    precision_dis = cm[1, 1] / cm[:, 1].sum() if cm[:, 1].sum() > 0 else 0

    _flush(f"\n{'='*60}")
    _flush(f"  Stage 1 — Binary Results (NOR vs DISEASE)")
    _flush(f"{'='*60}")
    _flush(f"  Overall accuracy : {acc*100:.1f}%")
    _flush(f"  NOR  recall      : {nor_recall*100:.1f}%  "
           f"({cm[0,0]}/20 correct)")
    _flush(f"  NOR  precision   : {precision_nor*100:.1f}%")
    _flush(f"  DIS  recall      : {dis_recall*100:.1f}%  "
           f"({cm[1,1]}/80 correct)")
    _flush(f"  DIS  precision   : {precision_dis*100:.1f}%")
    _flush(f"\n  Confusion matrix (rows=true, cols=pred):")
    _flush(f"            NOR  DISEASE")
    _flush(f"  NOR    : {cm[0,0]:>4}  {cm[0,1]:>4}")
    _flush(f"  DISEASE: {cm[1,0]:>4}  {cm[1,1]:>4}")

    # Which disease patients were misclassified as NOR?
    misclassified_as_nor = [
        (pid, IDX_TO_CLASS[orig])
        for pid, pred, orig in zip(pids, preds_b.tolist(), orig_labels)
        if pred == 0 and orig != CLASS_TO_IDX["NOR"]
    ]
    if misclassified_as_nor:
        _flush(f"\n  Disease patients misclassified as NOR ({len(misclassified_as_nor)}):")
        for pid, cls in misclassified_as_nor:
            _flush(f"    {pid}  ({cls})")

    _flush(f"{'='*60}\n")

    results = {
        "accuracy":         float(acc),
        "nor_recall":       float(nor_recall),
        "dis_recall":       float(dis_recall),
        "nor_precision":    float(precision_nor),
        "dis_precision":    float(precision_dis),
        "confusion_matrix": cm.tolist(),
    }

    # Save paired predictions — used by evaluate_pipeline.py
    preds_file = results_file.replace(".json", "_preds.json")
    with open(preds_file, "w") as f:
        json.dump({
            "pids":        pids,
            "preds_b":     preds_b.tolist(),      # binary: 0=NOR, 1=DISEASE
            "labels_b":    labels_b.tolist(),
            "orig_labels": orig_labels,            # original 5-class labels
            "accuracy":    float(acc),
        }, f, indent=2)
    _flush(f"Predictions saved → {preds_file}")
    _flush("(Pass this to evaluate_pipeline.py with --stage1_preds)\n")

    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--march9_dir",        type=str, required=True)
    p.add_argument("--training_dir",      type=str, required=True)
    p.add_argument("--n_patients",        type=int, default=100)
    p.add_argument("--epochs",            type=int, default=120)
    p.add_argument("--batch_size",        type=int, default=8)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--patience",          type=int, default=30)
    p.add_argument("--freeze_reg_epochs", type=int, default=20)
    p.add_argument("--n_splits",          type=int, default=5)
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--results_file",      type=str,
                   default="results/stage1_binary.json")
    p.add_argument("--wandb_project",     type=str, default=None)
    p.add_argument("--wandb_run_name",    type=str, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _flush(f"Device: {device}  |  Seed: {args.seed}")
    _flush(f"Task: Binary NOR vs DISEASE classification")
    _flush(f"Results → {args.results_file}\n")

    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    all_pids = discover_patients_from_images(args.march9_dir)[:args.n_patients]
    _flush(f"Using {len(all_pids)} patients: {all_pids[0]} … {all_pids[-1]}")

    results = stratified_kfold(
        args.march9_dir, args.training_dir, all_pids, args, device)

    results["config"] = vars(args)
    results["seed"]   = args.seed

    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    _flush(f"Results saved → {args.results_file}")


if __name__ == "__main__":
    main()
