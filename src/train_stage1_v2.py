"""
train_stage1_v2.py — Stage 1: Binary NOR vs DISEASE classifier (v2)
=====================================================================
Changes from v1 (train_stage1_binary.py):
  1. Checkpoint criterion changed from overall accuracy → balanced accuracy
     (average of NOR recall + DISEASE recall). This prevents the model from
     collapsing to "predict everything as DISEASE" which gave 80% overall
     accuracy but only 35% NOR recall.
  2. Added --model_type argument:
       full          → original GraphMotionEncoder + GRU pipeline (kept as-is)
       clinical_only → lightweight 3-layer MLP on clinical features only
                       (EF, volumes, mass — the features ACDC classes are
                        actually defined by). Trains in seconds, more stable
                        with only 20 NOR patients.
  3. Replaced weighted cross-entropy with focal loss (gamma=2). Focal loss
     down-weights easy DISEASE examples automatically, so you don't need to
     hand-tune class weights.
  4. Early stopping now monitors balanced accuracy (not val loss), consistent
     with the new checkpoint criterion.

Run (clinical_only — recommended first):
    mkdir -p results logs && nohup python train_stage1_v2.py \\
        --march9_dir  /home/amo/CINEMRI/data/ACDC/March9Data \\
        --training_dir /home/amo/CINEMRI/data/ACDC/training \\
        --n_patients 100 --epochs 120 --batch_size 8 \\
        --n_splits 5 --patience 30 --seed 42 \\
        --model_type clinical_only \\
        --results_file results/stage1_v2_clinical.json \\
        --wandb_project cardiac-ode --wandb_run_name stage1_v2_clinical \\
        > logs/stage1_v2_clinical.txt 2>&1 &

Run (full model with balanced checkpoint fix):
    nohup python train_stage1_v2.py \\
        --march9_dir  /home/amo/CINEMRI/data/ACDC/March9Data \\
        --training_dir /home/amo/CINEMRI/data/ACDC/training \\
        --n_patients 100 --epochs 120 --batch_size 8 \\
        --n_splits 5 --patience 30 --seed 42 \\
        --model_type full \\
        --results_file results/stage1_v2_full.json \\
        --wandb_project cardiac-ode --wandb_run_name stage1_v2_full \\
        > logs/stage1_v2_full.txt 2>&1 &

Output is identical to v1 (same _preds.json format) so evaluate_pipeline.py
works without changes.
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
BINARY_NAMES = {0: "NOR", 1: "DISEASE"}

def to_binary(label: int) -> int:
    return 0 if label == CLASS_TO_IDX["NOR"] else 1


# ── Focal loss ────────────────────────────────────────────────────────────────
# Replaces weighted cross-entropy. gamma=2 automatically focuses on hard
# examples (misclassified NOR patients) without manual weight tuning.

def focal_loss(logits: torch.Tensor,
               labels: torch.Tensor,
               gamma: float = 2.0,
               alpha: float = 0.85) -> torch.Tensor:
    """
    Binary focal loss.
    alpha: weight for the minority class (NOR=0). 0.85 means NOR is weighted
           ~6x more than DISEASE — necessary given 4:1 slice imbalance.
    gamma: focusing parameter. 2.0 is standard from Lin et al. (2017).
    """
    ce   = F.cross_entropy(logits, labels, reduction="none")
    pt   = torch.exp(-ce)
    # class-specific alpha weights
    at   = torch.where(labels == 0,
                       torch.tensor(alpha,     device=logits.device),
                       torch.tensor(1 - alpha, device=logits.device))
    loss = at * (1 - pt) ** gamma * ce
    return loss.mean()


# ── Model 1: Clinical-only MLP ────────────────────────────────────────────────
# Lightweight 3-layer MLP that takes clinical features only (EF, EDV, ESV,
# mass, etc.). This is essentially what Isensee et al. (ACDC 2017 winner) use
# for classification — the image pipeline is for segmentation; the classifier
# runs on derived clinical parameters.
#
# Pros: trains in seconds, no registration needed, very stable with 20 NOR
#       patients, directly uses the features ACDC classes are defined by.
# Cons: ignores motion patterns (fine for NOR vs DISEASE, less fine for
#       distinguishing HCM from NOR in edge cases).

class ClinicalOnlyStage1(nn.Module):
    def __init__(self, n_clinical: int = 8):
        super().__init__()
        # LayerNorm instead of BatchNorm1d — BatchNorm accumulates statistics
        # across the batch, which gets skewed when 80% of slices are DISEASE,
        # causing NOR features to be normalized toward DISEASE distributions.
        # LayerNorm normalizes per-sample so class imbalance doesn't affect it.
        self.net = nn.Sequential(
            nn.Linear(n_clinical, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self,
                clinical_feat: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict:
        logits = self.net(clinical_feat)
        loss   = torch.tensor(0.0, device=clinical_feat.device)
        if labels is not None:
            loss = focal_loss(logits, labels)
        return {"logits": logits, "total_loss": loss}


# ── Model 2: Full pipeline (original, kept intact) ────────────────────────────

@dataclass
class BinaryModelOutput:
    logits:     torch.Tensor
    total_loss: torch.Tensor
    cls_loss:   torch.Tensor
    reg_loss:   torch.Tensor
    bend_loss:  torch.Tensor
    fold_loss:  torch.Tensor


class BinaryClassifierHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.5):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class Stage1FullModel(nn.Module):
    """
    Original GraphMotionEncoder + GRU pipeline, unchanged from v1.
    Class weights replaced with focal loss.
    """

    def __init__(self,
                 d_z:         int   = 64,
                 n_verts:     int   = 64,
                 k_neighbors: int   = 6,
                 alpha:       float = 0.1,
                 beta:        float = 0.01,
                 n_clinical:  int   = 8):
        super().__init__()
        self.alpha      = alpha
        self.beta       = beta
        self.n_clinical = n_clinical

        self.registration  = RegistrationNet()
        self.reg_loss_fn   = RegistrationLoss(alpha=alpha, beta=beta)
        self.graph_encoder = GraphMotionEncoder(
            n_verts=n_verts, k=k_neighbors, out_dim=d_z)
        self.gru           = nn.GRU(input_size=d_z, hidden_size=d_z,
                                    num_layers=1, batch_first=True)
        self.classifier    = BinaryClassifierHead(
            in_dim=d_z + n_clinical, dropout=0.5)

    def forward(self,
                frames:        torch.Tensor,
                masks:         torch.Tensor,
                times:         torch.Tensor,
                labels:        Optional[torch.Tensor] = None,
                clinical_feat: Optional[torch.Tensor] = None) -> BinaryModelOutput:

        B, N_frames, C, H, W = frames.shape
        N_pairs  = N_frames - 1
        device   = frames.device

        all_embeddings  = []
        total_reg_loss  = torch.tensor(0.0, device=device)
        total_bend_loss = torch.tensor(0.0, device=device)
        total_fold_loss = torch.tensor(0.0, device=device)

        for t in range(N_pairs):
            warped, vel_field, phi = self.registration.get_warped(
                frames[:, t], frames[:, t+1])
            reg_total, reg_ncc, reg_bend, reg_fold = self.reg_loss_fn(
                warped, frames[:, t], vel_field, phi, masks[:, t])
            total_reg_loss  = total_reg_loss  + reg_ncc
            total_bend_loss = total_bend_loss + reg_bend
            total_fold_loss = total_fold_loss + reg_fold
            all_embeddings.append(self.graph_encoder(masks[:, t], phi))

        total_reg_loss  = total_reg_loss  / N_pairs
        total_bend_loss = total_bend_loss / N_pairs
        total_fold_loss = total_fold_loss / N_pairs

        embeddings_seq = torch.stack(all_embeddings, dim=1)
        _, h_n         = self.gru(embeddings_seq)
        z0             = h_n.squeeze(0)

        if clinical_feat is not None and self.n_clinical > 0:
            z_in = torch.cat([z0, clinical_feat.to(device)], dim=1)
        else:
            z_in = torch.cat([z0, torch.zeros(z0.shape[0], self.n_clinical,
                                               device=device)], dim=1)

        logits = self.classifier(z_in)

        cls_loss = torch.tensor(0.0, device=device)
        if labels is not None:
            # v2: focal loss instead of weighted cross-entropy
            cls_loss = focal_loss(logits, labels, gamma=2.0, alpha=0.75)

        total_loss = (cls_loss
                      + self.alpha * total_reg_loss
                      + self.beta  * (total_bend_loss + total_fold_loss))

        return BinaryModelOutput(
            logits=logits, total_loss=total_loss, cls_loss=cls_loss,
            reg_loss=total_reg_loss, bend_loss=total_bend_loss,
            fold_loss=total_fold_loss)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _flush(*args): print(*args, flush=True)

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


# ── Training loops ────────────────────────────────────────────────────────────

def train_epoch_clinical(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    n = 0

    for batch in loader:
        clin     = batch["clinical_feat"].to(device)
        labels_5 = batch["label"].to(device)
        labels_b = torch.tensor([to_binary(l.item()) for l in labels_5],
                                 device=device)

        optimizer.zero_grad()
        out = model(clin, labels_b)
        out["total_loss"].backward()
        optimizer.step()

        total_loss += out["total_loss"].item()
        preds = out["logits"].argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels_b.cpu().tolist())
        n += 1

    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return {"total_loss": total_loss / max(n, 1), "accuracy": acc}


def train_epoch_full(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    n = 0

    for batch in loader:
        frames   = batch["frames"].to(device)
        masks    = batch["masks"].to(device)
        times    = batch["times"].to(device)
        clin     = batch["clinical_feat"].to(device)
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
    return {"total_loss": total_loss / max(n, 1), "accuracy": acc}


@torch.no_grad()
def val_epoch(model, loader, device, model_type: str):
    model.eval()
    total_loss = 0.0
    n = 0
    patient_data:       Dict[str, list] = defaultdict(list)
    patient_label_map:  Dict[str, int]  = {}
    patient_orig_label: Dict[str, int]  = {}

    for batch in loader:
        clin     = batch["clinical_feat"].to(device)
        labels_5 = batch["label"].to(device)
        labels_b = torch.tensor([to_binary(l.item()) for l in labels_5],
                                 device=device)

        if model_type == "clinical_only":
            out        = model(clin, labels_b)
            logits_cpu = out["logits"].detach().cpu()
            total_loss += out["total_loss"].item()
        else:
            frames = batch["frames"].to(device)
            masks  = batch["masks"].to(device)
            times  = batch["times"].to(device)
            out    = model(frames, masks, times, labels_b, clinical_feat=clin)
            logits_cpu = out.logits.detach().cpu()
            total_loss += out.total_loss.item()

        n += 1
        for i, meta in enumerate(batch["meta"]):
            pid = meta["patient_id"]
            w   = float(meta["n_myo_pixels"])
            patient_data[pid].append((logits_cpu[i], w))
            patient_label_map[pid]  = labels_b[i].item()
            patient_orig_label[pid] = labels_5[i].item()

    all_preds_b, all_labels_b = [], []
    all_pids,    all_orig     = [], []

    for pid, data in patient_data.items():
        logit_stack = torch.stack([d[0] for d in data])
        weights     = torch.tensor([d[1] for d in data])
        weights     = weights / (weights.sum() + 1e-8)
        mean_logit  = (logit_stack * weights.unsqueeze(1)).sum(0)
        pred_b      = int(mean_logit.argmax().item())

        all_preds_b.append(pred_b)
        all_labels_b.append(patient_label_map[pid])
        all_pids.append(pid)
        all_orig.append(patient_orig_label[pid])

    preds_b  = np.array(all_preds_b)
    labels_b = np.array(all_labels_b)
    acc      = (preds_b == labels_b).mean()

    nor_mask   = labels_b == 0
    nor_recall = (preds_b[nor_mask] == 0).mean() if nor_mask.sum() > 0 else 0.0
    dis_recall = (preds_b[~nor_mask] == 1).mean() if (~nor_mask).sum() > 0 else 0.0

    # v2: balanced accuracy = mean of per-class recall
    balanced_acc = (nor_recall + dis_recall) / 2.0

    return {
        "total_loss":    total_loss / max(n, 1),
        "accuracy":      float(acc),
        "balanced_acc":  float(balanced_acc),   # ← new primary metric
        "nor_recall":    float(nor_recall),
        "dis_recall":    float(dis_recall),
        "preds_b":       all_preds_b,
        "labels_b":      all_labels_b,
        "pids":          all_pids,
        "orig_labels":   all_orig,
    }


# ── K-fold CV ─────────────────────────────────────────────────────────────────

def stratified_kfold(march9_dir, training_dir, patient_ids, args, device):
    labels_all = []
    for pid in patient_ids:
        info = parse_patient_info(training_dir, pid)
        labels_all.append(CLASS_TO_IDX.get(info["group"], 0))

    _flush(f"\nUsing {len(patient_ids)} patients: {patient_ids[0]} … {patient_ids[-1]}")
    _flush(f"Model type: {args.model_type}")

    skf = StratifiedKFold(n_splits=args.n_splits,
                          shuffle=True, random_state=args.seed)

    all_preds_b, all_labels_b = [], []
    all_pids,    all_orig     = [], []

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
                                  shuffle=True,  collate_fn=collate_fn,
                                  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate_fn,
                                  num_workers=2, pin_memory=True)

        # ── Build model ───────────────────────────────────────────────────────
        if args.model_type == "clinical_only":
            model = ClinicalOnlyStage1(
                n_clinical=N_CLINICAL_FEATURES).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=1e-3)
        else:
            model = Stage1FullModel(n_clinical=N_CLINICAL_FEATURES).to(device)
            freeze_registration(model)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=5e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)

        use_wandb = WANDB_AVAILABLE and getattr(args, "wandb_project", None)
        if use_wandb:
            wandb.init(project=args.wandb_project,
                       name=f"{args.wandb_run_name}_fold{fold_idx+1}",
                       config=vars(args), reinit=True,
                       tags=["stage1", "v2", args.model_type])

        # ── v2: checkpoint on balanced accuracy, not overall accuracy ─────────
        best_balanced  = 0.0
        best_val_data  = {}
        patience_ctr   = 0
        reg_unfrozen   = False

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            if (args.model_type == "full"
                    and not reg_unfrozen
                    and epoch > args.freeze_reg_epochs):
                unfreeze_registration(model, optimizer, args.lr)
                reg_unfrozen = True

            if args.model_type == "clinical_only":
                train_m = train_epoch_clinical(model, train_loader,
                                               optimizer, device)
            else:
                train_m = train_epoch_full(model, train_loader,
                                           optimizer, device)

            val_m = val_epoch(model, val_loader, device, args.model_type)
            scheduler.step()

            _flush(
                f"  [Fold {fold_idx+1} | Ep {epoch:3d}/{args.epochs}] "
                f"loss={train_m['total_loss']:.3f} "
                f"train_acc={train_m['accuracy']*100:.1f}% "
                f"val_acc={val_m['accuracy']*100:.1f}% "
                f"bal_acc={val_m['balanced_acc']*100:.1f}% "
                f"NOR={val_m['nor_recall']*100:.1f}% "
                f"DIS={val_m['dis_recall']*100:.1f}% "
                f"pat={patience_ctr}/{args.patience} "
                f"({time.time()-t0:.0f}s)"
            )

            if use_wandb:
                wandb.log({
                    "epoch":             epoch,
                    "train/loss":        train_m["total_loss"],
                    "train/acc":         train_m["accuracy"],
                    "val/acc":           val_m["accuracy"],
                    "val/balanced_acc":  val_m["balanced_acc"],
                    "val/nor_recall":    val_m["nor_recall"],
                    "val/dis_recall":    val_m["dis_recall"],
                    "val/loss":          val_m["total_loss"],
                    "patience":          patience_ctr,
                    "lr":                scheduler.get_last_lr()[0],
                })

            # ── v2: save best by balanced accuracy ────────────────────────────
            if val_m["balanced_acc"] > best_balanced:
                best_balanced = val_m["balanced_acc"]
                best_val_data = val_m
                patience_ctr  = 0          # reset patience on improvement
            else:
                patience_ctr += 1

            if patience_ctr >= args.patience:
                _flush(f"  Early stopping at epoch {epoch}")
                break

        if use_wandb:
            wandb.finish()

        _flush(f"  Best balanced acc: {best_balanced*100:.1f}%  "
               f"(NOR {best_val_data.get('nor_recall',0)*100:.1f}% / "
               f"DIS {best_val_data.get('dis_recall',0)*100:.1f}%)")

        all_preds_b.extend(best_val_data.get("preds_b",  []))
        all_labels_b.extend(best_val_data.get("labels_b", []))
        all_pids.extend(best_val_data.get("pids",        []))
        all_orig.extend(best_val_data.get("orig_labels", []))

    return _summarize(all_preds_b, all_labels_b,
                      all_pids, all_orig,
                      args.model_type, args.results_file)


# ── Summary ───────────────────────────────────────────────────────────────────

def _summarize(preds_b, labels_b, pids, orig_labels, model_type, results_file):
    preds_b  = np.array(preds_b)
    labels_b = np.array(labels_b)
    acc      = (preds_b == labels_b).mean()
    cm       = confusion_matrix(labels_b, preds_b, labels=[0, 1])

    nor_recall    = cm[0, 0] / cm[0].sum()    if cm[0].sum()    > 0 else 0
    dis_recall    = cm[1, 1] / cm[1].sum()    if cm[1].sum()    > 0 else 0
    precision_nor = cm[0, 0] / cm[:, 0].sum() if cm[:, 0].sum() > 0 else 0
    precision_dis = cm[1, 1] / cm[:, 1].sum() if cm[:, 1].sum() > 0 else 0
    balanced_acc  = (nor_recall + dis_recall) / 2.0

    _flush(f"\n{'='*60}")
    _flush(f"  Stage 1 v2 ({model_type}) — Binary Results (NOR vs DISEASE)")
    _flush(f"{'='*60}")
    _flush(f"  Overall accuracy  : {acc*100:.1f}%")
    _flush(f"  Balanced accuracy : {balanced_acc*100:.1f}%  ← primary metric")
    _flush(f"  NOR  recall       : {nor_recall*100:.1f}%  "
           f"({cm[0,0]}/20 correct)")
    _flush(f"  NOR  precision    : {precision_nor*100:.1f}%")
    _flush(f"  DIS  recall       : {dis_recall*100:.1f}%  "
           f"({cm[1,1]}/80 correct)")
    _flush(f"  DIS  precision    : {precision_dis*100:.1f}%")
    _flush(f"\n  Confusion matrix (rows=true, cols=pred):")
    _flush(f"            NOR  DISEASE")
    _flush(f"  NOR    : {cm[0,0]:>4}  {cm[0,1]:>4}")
    _flush(f"  DISEASE: {cm[1,0]:>4}  {cm[1,1]:>4}")

    misclassified = [
        (pid, IDX_TO_CLASS[orig])
        for pid, pred, orig in zip(pids, preds_b.tolist(), orig_labels)
        if pred == 0 and orig != CLASS_TO_IDX["NOR"]
    ]
    if misclassified:
        _flush(f"\n  Disease patients misclassified as NOR ({len(misclassified)}):")
        for pid, cls in misclassified:
            _flush(f"    {pid}  ({cls})")

    _flush(f"{'='*60}\n")

    results = {
        "accuracy":         float(acc),
        "balanced_acc":     float(balanced_acc),
        "nor_recall":       float(nor_recall),
        "dis_recall":       float(dis_recall),
        "nor_precision":    float(precision_nor),
        "dis_precision":    float(precision_dis),
        "confusion_matrix": cm.tolist(),
        "model_type":       model_type,
    }

    # Same output format as v1 — evaluate_pipeline.py needs no changes
    preds_file = results_file.replace(".json", "_preds.json")
    with open(preds_file, "w") as f:
        json.dump({
            "pids":        pids,
            "preds_b":     preds_b.tolist(),
            "labels_b":    labels_b.tolist(),
            "orig_labels": orig_labels,
            "accuracy":    float(acc),
            "balanced_acc": float(balanced_acc),
        }, f, indent=2)
    _flush(f"Predictions saved → {preds_file}")
    _flush("(Pass this to evaluate_pipeline.py with --stage1_preds)\n")

    return results


# ── Args ──────────────────────────────────────────────────────────────────────

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
    p.add_argument("--model_type",        type=str, default="clinical_only",
                   choices=["clinical_only", "full"])
    p.add_argument("--results_file",      type=str,
                   default="results/stage1_v2.json")
    p.add_argument("--wandb_project",     type=str, default=None)
    p.add_argument("--wandb_run_name",    type=str, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _flush(f"Device: {device}  |  Seed: {args.seed}")
    _flush(f"Task: Binary NOR vs DISEASE  |  Model: {args.model_type}")
    _flush(f"Results → {args.results_file}\n")

    os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    all_pids = discover_patients_from_images(args.march9_dir)[:args.n_patients]

    results = stratified_kfold(
        args.march9_dir, args.training_dir, all_pids, args, device)

    results["config"] = vars(args)
    results["seed"]   = args.seed

    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    _flush(f"Results saved → {args.results_file}")


if __name__ == "__main__":
    main()
