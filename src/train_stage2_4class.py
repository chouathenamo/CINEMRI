"""
train_stage2_4class.py — Stage 2: 4-class classifier (DCM/HCM/MINF/RV only)
=============================================================================
Trains on disease patients ONLY (NOR excluded from all folds).
Supports both no-ODE and idea3 via --model_variant argument.

The model now has 4 classes: DCM=0, HCM=1, MINF=2, RV=3
(NOR is dropped entirely from training and evaluation).

Run (no-ODE variant):
    mkdir -p results logs && nohup python train_stage2_4class.py \\
        --march9_dir  /home/amo/CINEMRI/data/ACDC/March9Data \\
        --training_dir /home/amo/CINEMRI/data/ACDC/training \\
        --n_patients 100 --epochs 120 --batch_size 8 \\
        --n_splits 5 --patience 30 --seed 42 \\
        --model_variant no_ode \\
        --results_file results/stage2_no_ode.json \\
        --wandb_project cardiac-ode --wandb_run_name stage2_no_ode \\
        > logs/stage2_no_ode.txt 2>&1 &

Run (idea3 variant):
    nohup python train_stage2_4class.py \\
        ... --model_variant idea3 \\
        --results_file results/stage2_idea3.json \\
        --wandb_run_name stage2_idea3 \\
        > logs/stage2_idea3.txt 2>&1 &
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


# ── 4-class label mapping (NOR excluded) ─────────────────────────────────────
# Original: NOR=0, DCM=1, HCM=2, MINF=3, RV=4
# Stage 2:          DCM=0, HCM=1, MINF=2, RV=3
DISEASE_CLASSES   = ["DCM", "HCM", "MINF", "RV"]
ORIG_TO_STAGE2    = {CLASS_TO_IDX[c]: i for i, c in enumerate(DISEASE_CLASSES)}
STAGE2_TO_NAME    = {i: c for i, c in enumerate(DISEASE_CLASSES)}
NOR_IDX           = CLASS_TO_IDX["NOR"]


def is_disease(label: int) -> bool:
    return label != NOR_IDX


def to_stage2_label(orig_label: int) -> int:
    return ORIG_TO_STAGE2[orig_label]


# ── ODE function (small, idea3 variant) ──────────────────────────────────────

class ODEFunc(nn.Module):
    """Small ODE (idea3): hidden_dim=32, single layer."""
    def __init__(self, d_z=64, hidden_dim=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z + 1, hidden_dim),
            nn.Tanh(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_z),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        return self.net(torch.cat([z, t.expand(z.shape[0], 1)], dim=1))

    def reset_nfe(self): self.nfe = 0


class NeuralODEBlock(nn.Module):
    def __init__(self, d_z=64, method='euler'):
        super().__init__()
        self.method  = method
        self.odefunc = ODEFunc(d_z=d_z)

    def forward(self, z0, t_span):
        from torchdiffeq import odeint
        self.odefunc.reset_nfe()
        return odeint(self.odefunc, z0, t_span,
                      method=self.method, rtol=1e-3, atol=1e-4)


# ── Model output ──────────────────────────────────────────────────────────────

@dataclass
class Stage2Output:
    logits:     torch.Tensor
    total_loss: torch.Tensor
    cls_loss:   torch.Tensor
    reg_loss:   torch.Tensor
    bend_loss:  torch.Tensor
    fold_loss:  torch.Tensor


# ── Stage 2 model ─────────────────────────────────────────────────────────────

class Stage2Model(nn.Module):
    """
    4-class classifier for DCM / HCM / MINF / RV.
    model_variant = 'no_ode' or 'idea3'
    """

    def __init__(
        self,
        d_z:           int   = 64,
        n_verts:       int   = 64,
        k_neighbors:   int   = 6,
        alpha:         float = 0.1,
        beta:          float = 0.01,
        n_clinical:    int   = 8,
        model_variant: str   = 'no_ode',   # 'no_ode' or 'idea3'
    ):
        super().__init__()
        self.alpha         = alpha
        self.beta          = beta
        self.n_clinical    = n_clinical
        self.model_variant = model_variant

        self.registration  = RegistrationNet()
        self.reg_loss_fn   = RegistrationLoss(alpha=alpha, beta=beta)
        self.graph_encoder = GraphMotionEncoder(
            n_verts=n_verts, k=k_neighbors, out_dim=d_z)
        self.gru           = nn.GRU(input_size=d_z, hidden_size=d_z,
                                    num_layers=1, batch_first=True)

        if model_variant == 'idea3':
            self.ode_block = NeuralODEBlock(d_z=d_z, method='euler')

        # Classifier: 4 output classes
        # class weights: upweight MINF (hardest to distinguish)
        self.classifier = nn.Sequential(
            nn.Linear(d_z + n_clinical, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 4),
        )

    def forward(
        self,
        frames:        torch.Tensor,
        masks:         torch.Tensor,
        times:         torch.Tensor,
        labels:        Optional[torch.Tensor] = None,   # stage2 labels (0-3)
        clinical_feat: Optional[torch.Tensor] = None,
    ) -> Stage2Output:

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
        _, h_n = self.gru(embeddings_seq)
        z0     = h_n.squeeze(0)                    # (B, d_z)

        if self.model_variant == 'idea3':
            # Small ODE + mean-pool + residual
            times_mid = (times[:, :-1] + times[:, 1:]) / 2.0
            t_span    = times_mid[0].to(device)
            t_span, _ = torch.sort(t_span)
            eps = 1e-5
            for i in range(1, len(t_span)):
                if t_span[i] <= t_span[i-1]:
                    t_span[i] = t_span[i-1] + eps
            z_traj = self.ode_block(z0, t_span)    # (T, B, d_z)
            z_final = z0 + z_traj.mean(dim=0)
        else:
            # no_ode: just use GRU output
            z_final = z0

        if clinical_feat is not None and self.n_clinical > 0:
            z_in = torch.cat([z_final, clinical_feat.to(device)], dim=1)
        else:
            pad  = torch.zeros(z_final.shape[0], self.n_clinical, device=device)
            z_in = torch.cat([z_final, pad], dim=1)

        logits = self.classifier(z_in)

        if labels is not None:
            # Up-weight MINF (hardest confusable pair)
            weight = torch.ones(4, device=device)
            weight[ORIG_TO_STAGE2[CLASS_TO_IDX["MINF"]]] = 2.0
            cls_loss = F.cross_entropy(logits, labels, weight=weight,
                                       label_smoothing=0.1)
        else:
            cls_loss = torch.tensor(0.0, device=device)

        total_loss = (cls_loss
                      + self.alpha * total_reg_loss
                      + self.beta  * (total_bend_loss + total_fold_loss))

        return Stage2Output(
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
        frames   = batch["frames"].to(device)
        masks    = batch["masks"].to(device)
        times    = batch["times"].to(device)
        clin     = batch["clinical_feat"].to(device)
        labels_5 = batch["label"].to(device)

        # Convert to stage2 labels (0-3)
        labels_s2 = torch.tensor(
            [to_stage2_label(l.item()) for l in labels_5], device=device)

        optimizer.zero_grad()
        out = model(frames, masks, times, labels_s2, clinical_feat=clin)
        out.total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += out.total_loss.item()
        preds = out.logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels_s2.cpu().tolist())
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
    patient_orig_map:  Dict[str, int]  = {}

    for batch in loader:
        frames   = batch["frames"].to(device)
        masks    = batch["masks"].to(device)
        times    = batch["times"].to(device)
        clin     = batch["clinical_feat"].to(device)
        labels_5 = batch["label"].to(device)
        labels_s2 = torch.tensor(
            [to_stage2_label(l.item()) for l in labels_5], device=device)

        out        = model(frames, masks, times, labels_s2, clinical_feat=clin)
        total_loss += out.total_loss.item()
        n          += 1

        logits_cpu = out.logits.detach().cpu()
        for i, meta in enumerate(batch["meta"]):
            pid = meta["patient_id"]
            w   = float(meta["n_myo_pixels"])
            patient_data[pid].append((logits_cpu[i], w))
            patient_label_map[pid] = labels_s2[i].item()
            patient_orig_map[pid]  = labels_5[i].item()

    all_preds, all_labels = [], []
    all_pids, all_orig    = [], []

    for pid, data in patient_data.items():
        logit_stack = torch.stack([d[0] for d in data])
        weights     = torch.tensor([d[1] for d in data])
        weights     = weights / (weights.sum() + 1e-8)
        mean_logit  = (logit_stack * weights.unsqueeze(1)).sum(0)
        all_preds.append(int(mean_logit.argmax().item()))
        all_labels.append(patient_label_map[pid])
        all_pids.append(pid)
        all_orig.append(patient_orig_map[pid])

    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return {
        "total_loss": total_loss / max(n, 1),
        "accuracy":   float(acc),
        "preds":      all_preds,
        "labels":     all_labels,
        "pids":       all_pids,
        "orig":       all_orig,
    }


# ── K-fold CV (disease patients only) ────────────────────────────────────────

def stratified_kfold(march9_dir, training_dir, patient_ids, args, device):
    # Filter to disease patients only
    disease_pids    = []
    disease_labels  = []
    for pid in patient_ids:
        info = parse_patient_info(training_dir, pid)
        orig = CLASS_TO_IDX.get(info["group"], 0)
        if is_disease(orig):
            disease_pids.append(pid)
            disease_labels.append(orig)

    _flush(f"Disease patients: {len(disease_pids)} "
           f"(NOR excluded from all folds)")
    for cls in DISEASE_CLASSES:
        n = sum(1 for l in disease_labels if l == CLASS_TO_IDX[cls])
        _flush(f"  {cls}: {n}")

    skf = StratifiedKFold(n_splits=args.n_splits,
                          shuffle=True, random_state=args.seed)

    all_preds, all_labels = [], []
    all_pids, all_orig    = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(disease_pids, disease_labels)):

        train_pids = [disease_pids[i] for i in train_idx]
        val_pids   = [disease_pids[i] for i in val_idx]

        _flush(f"\nFold {fold_idx+1}/{args.n_splits}: "
               f"{len(train_pids)} train / {len(val_pids)} val  "
               f"(disease only, seed={args.seed})")

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

        model = Stage2Model(
            n_clinical=N_CLINICAL_FEATURES,
            model_variant=args.model_variant,
        ).to(device)
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
                       tags=["stage2", "4class", args.model_variant])

        best_val_acc  = 0.0
        best_val_loss = float("inf")
        best_val_data = {}
        patience_ctr  = 0
        reg_unfrozen  = False

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
                f"val_loss={val_m['total_loss']:.3f} "
                f"pat={patience_ctr}/{args.patience} "
                f"({time.time()-t0:.0f}s)"
            )

            if use_wandb:
                wandb.log({
                    "epoch":      epoch,
                    "train/loss": train_m["total_loss"],
                    "train/acc":  train_m["accuracy"],
                    "val/acc":    val_m["accuracy"],
                    "val/loss":   val_m["total_loss"],
                    "patience":   patience_ctr,
                    "lr":         scheduler.get_last_lr()[0],
                })

            if val_m["accuracy"] > best_val_acc:
                best_val_acc  = val_m["accuracy"]
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

        _flush(f"  Best val acc: {best_val_acc*100:.1f}%")
        all_preds.extend(best_val_data.get("preds", []))
        all_labels.extend(best_val_data.get("labels", []))
        all_pids.extend(best_val_data.get("pids", []))
        all_orig.extend(best_val_data.get("orig", []))

    return _summarize(all_preds, all_labels, all_pids, all_orig,
                      args.model_variant, args.results_file)


def _summarize(preds, labels, pids, orig_labels, variant, results_file):
    preds  = np.array(preds)
    labels = np.array(labels)
    acc    = (preds == labels).mean()
    cm     = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])

    _flush(f"\n{'='*60}")
    _flush(f"  Stage 2 ({variant}) — 4-class Results")
    _flush(f"{'='*60}")
    _flush(f"  Overall accuracy: {acc*100:.1f}%  "
           f"({(preds==labels).sum()}/{len(labels)} disease patients)")
    _flush(f"\n  Per-class:")
    for i, cls in enumerate(DISEASE_CLASSES):
        mask = labels == i
        if mask.sum() > 0:
            ca = (preds[mask] == i).mean()
            _flush(f"    {cls}: {ca*100:.1f}%  ({mask.sum()} patients)")
    _flush(f"\n  Confusion matrix (rows=true, cols=pred):")
    header = "       " + "  ".join(f"{c:>5}" for c in DISEASE_CLASSES)
    _flush(header)
    for i, row in enumerate(cm):
        _flush(f"  {DISEASE_CLASSES[i]:<5}: " +
               "  ".join(f"{v:>5}" for v in row))
    _flush(f"{'='*60}\n")

    results = {
        "accuracy":         float(acc),
        "model_variant":    variant,
        "confusion_matrix": cm.tolist(),
        "per_class":        {},
    }

    # Save paired predictions for evaluate_pipeline.py
    preds_file = results_file.replace(".json", "_preds.json")
    with open(preds_file, "w") as f:
        json.dump({
            "pids":        pids,
            "preds_s2":    preds.tolist(),     # stage2 labels (0-3)
            "labels_s2":   labels.tolist(),
            "orig_labels": orig_labels,        # original 5-class labels
            "accuracy":    float(acc),
            "variant":     variant,
        }, f, indent=2)
    _flush(f"Predictions saved → {preds_file}")

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
    p.add_argument("--model_variant",     type=str, default="no_ode",
                   choices=["no_ode", "idea3"])
    p.add_argument("--results_file",      type=str,
                   default="results/stage2_no_ode.json")
    p.add_argument("--wandb_project",     type=str, default=None)
    p.add_argument("--wandb_run_name",    type=str, default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _flush(f"Device: {device}  |  Seed: {args.seed}")
    _flush(f"Model variant: {args.model_variant}")
    _flush(f"Task: 4-class DCM/HCM/MINF/RV (NOR excluded)")
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
