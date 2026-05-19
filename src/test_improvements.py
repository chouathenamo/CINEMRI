"""
test_improvements.py — Fast ablation of NeurIPS/MICCAI-style improvements
==========================================================================
Tests several drop-in improvements to train_stage2_4class.py (4-class
DCM/HCM/MINF/RV classification on ACDC CINE MRI) and prints a single
comparison table.

Each experiment is one Stage-2 K-fold CV run with a small variation on the
baseline.  Use --quick for a smoke run (few patients, few epochs); use the
default args for a real ablation.

Improvements & paper references
-------------------------------
  baseline       Current GRU + weighted CE (train_stage2_4class.py).
  focal_loss     Focal loss γ=2 — Lin et al., "Focal Loss for Dense Object
                 Detection", ICCV 2017.  Standard fix for class imbalance,
                 used in many cardiac classifiers (e.g. Zheng et al., MICCAI
                 2019).
  bidir_gru      Bidirectional GRU — captures both systolic and diastolic
                 motion (Schuster & Paliwal 1997; widely adopted in cardiac
                 motion papers, e.g. Qin et al., MICCAI 2018).
  transformer    Replace GRU with a 2-layer TransformerEncoder — see
                 Vaswani et al. NeurIPS 2017; cardiac variants in TransUNet
                 (Chen et al. 2021) and Reynaud et al. (MICCAI 2021).
  warmup_cosine  Linear warmup + cosine LR — ViT recipe (Dosovitskiy et al.,
                 ICLR 2021); typically +1-2 pts over plain cosine on small
                 medical datasets.
  strong_aug     Elastic deformation + Gaussian noise — nnU-Net's standard
                 augmentation recipe (Isensee et al., Nature Methods 2021;
                 NeurIPS 2019 ACDC winner).
  larger_model   d_z=128, deeper classifier — capacity-scaling baseline.
  conf_pool      Entropy-weighted slice aggregation instead of myo-pixel
                 weighting (MIL intuition, Ilse et al., ICML 2018).  Zero
                 extra params; reweights by per-slice prediction confidence.
  clinical_only  MLP on the 8 clinical features only — ablation showing how
                 much motion adds over EF/EDV/ESV (cf. Diller et al. 2019,
                 Wolterink et al. 2017).
  best_combo     focal_loss + bidir_gru + warmup_cosine + conf_pool.

Usage
-----
  # Quick smoke run (≈3 min):
  python test_improvements.py --quick --march9_dir <...> --training_dir <...>

  # Full ablation (slower, comparable to one train_stage2_4class.py run
  # per experiment):
  python test_improvements.py --march9_dir <...> --training_dir <...> \\
        --epochs 40 --n_splits 3

  # Run a subset:
  python test_improvements.py --experiments focal_loss bidir_gru ...
"""

import os, sys, json, time, math, argparse, copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_v3 import (ACDCSliceDataset, collate_fn, CLASS_TO_IDX,
                         discover_patients_from_images, parse_patient_info,
                         N_CLINICAL_FEATURES)
from registration import RegistrationNet, RegistrationLoss
from graph_v3    import GraphMotionEncoder

DISEASE_CLASSES = ["DCM", "HCM", "MINF", "RV"]
ORIG_TO_STAGE2  = {CLASS_TO_IDX[c]: i for i, c in enumerate(DISEASE_CLASSES)}
NOR_IDX         = CLASS_TO_IDX["NOR"]
MINF_S2_IDX     = ORIG_TO_STAGE2[CLASS_TO_IDX["MINF"]]


def _flush(*a): print(*a, flush=True)


# ── Losses ───────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss (Lin et al. ICCV 2017) with optional class weights."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.ls = label_smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        p    = logp.exp()
        if self.ls > 0:
            n_cls = logits.size(-1)
            true_dist = torch.full_like(logp, self.ls / (n_cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.ls)
            ce = -(true_dist * logp).sum(-1)
            pt = (true_dist * p).sum(-1)
        else:
            ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
            pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        if self.weight is not None and self.ls > 0:
            w = self.weight[target]
            ce = ce * w
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Schedulers ───────────────────────────────────────────────────────────────

def make_scheduler(name, optimizer, epochs, warmup=5):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)
    if name == "warmup_cosine":
        def lr_lambda(ep):
            if ep < warmup:
                return (ep + 1) / max(1, warmup)
            progress = (ep - warmup) / max(1, epochs - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    raise ValueError(name)


# ── Stronger augmentation (elastic deform + Gaussian noise) ──────────────────

class ElasticAugDataset(ACDCSliceDataset):
    """Adds nnU-Net-style elastic deformation + Gaussian noise on top of the
    base flips/rotation/jitter pipeline."""

    def _augment(self, frames_arr, masks_arr):
        import cv2, random
        frames_arr, masks_arr = super()._augment(frames_arr, masks_arr)
        N, H, W = frames_arr.shape

        # Elastic deformation: random low-frequency displacement field.
        if random.random() < 0.5:
            alpha = random.uniform(20.0, 40.0)   # displacement magnitude (px)
            sigma = random.uniform(6.0, 8.0)     # smoothing kernel
            dx = cv2.GaussianBlur(
                (np.random.rand(H, W).astype(np.float32) * 2 - 1),
                (0, 0), sigma) * alpha
            dy = cv2.GaussianBlur(
                (np.random.rand(H, W).astype(np.float32) * 2 - 1),
                (0, 0), sigma) * alpha
            map_x = (np.arange(W)[None, :] + dx).astype(np.float32)
            map_y = (np.arange(H)[:, None] + dy).astype(np.float32)
            for i in range(N):
                frames_arr[i] = cv2.remap(
                    frames_arr[i], map_x, map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101)
                masks_arr[i] = cv2.remap(
                    masks_arr[i], map_x, map_y,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Additive Gaussian noise on frames only.
        if random.random() < 0.5:
            frames_arr = np.clip(
                frames_arr + np.random.randn(*frames_arr.shape).astype(
                    np.float32) * 0.03, 0.0, 1.0)
        return frames_arr, masks_arr


# ── Improved Stage-2 model ───────────────────────────────────────────────────

@dataclass
class ExpConfig:
    name:            str   = "baseline"
    # architecture
    d_z:             int   = 64
    bidir_gru:       bool  = False
    temporal:        str   = "gru"          # 'gru' | 'transformer'
    deeper_clf:      bool  = False
    # loss
    loss:            str   = "ce"           # 'ce' | 'focal'
    focal_gamma:     float = 2.0
    minf_weight:     float = 2.0
    label_smooth:    float = 0.1
    # training
    schedule:        str   = "cosine"       # 'cosine' | 'warmup_cosine'
    warmup_epochs:   int   = 5
    strong_aug:      bool  = False
    # validation
    pool:            str   = "myo"          # 'myo' | 'entropy'
    # specials
    clinical_only:   bool  = False
    # registration loss weights (kept = baseline by default)
    alpha:           float = 0.1
    beta:            float = 0.01


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):       # x: (B, T, d)
        return x + self.pe[:, :x.size(1)]


class ClinicalOnlyModel(nn.Module):
    """Ablation: MLP on the 8 clinical features only (no motion)."""
    def __init__(self, n_clinical=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_clinical, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 4),
        )

    def forward(self, frames, masks, times, labels=None, clinical_feat=None):
        logits = self.net(clinical_feat)
        return _wrap_out(logits, frames.device)


@dataclass
class Out:
    logits: torch.Tensor
    total_loss: torch.Tensor
    cls_loss:   torch.Tensor
    reg_loss:   torch.Tensor
    bend_loss:  torch.Tensor
    fold_loss:  torch.Tensor


def _wrap_out(logits, device):
    z = torch.tensor(0.0, device=device)
    return Out(logits=logits, total_loss=z.clone(), cls_loss=z.clone(),
               reg_loss=z, bend_loss=z, fold_loss=z)


class ImprovedStage2Model(nn.Module):
    """Stage-2 model wrapped behind an ExpConfig so we can flip improvements
    one at a time without touching train_stage2_4class.py."""

    def __init__(self, cfg: ExpConfig, n_clinical=N_CLINICAL_FEATURES):
        super().__init__()
        self.cfg = cfg
        d_z = cfg.d_z

        self.registration  = RegistrationNet()
        self.reg_loss_fn   = RegistrationLoss(alpha=cfg.alpha, beta=cfg.beta)
        self.graph_encoder = GraphMotionEncoder(out_dim=d_z)

        if cfg.temporal == "gru":
            self.gru = nn.GRU(d_z, d_z, batch_first=True,
                              bidirectional=cfg.bidir_gru)
            gru_out = d_z * (2 if cfg.bidir_gru else 1)
            self.temporal_proj = (nn.Linear(gru_out, d_z)
                                  if cfg.bidir_gru else nn.Identity())
        elif cfg.temporal == "transformer":
            self.pos = PositionalEncoding(d_z, max_len=64)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_z, nhead=4, dim_feedforward=128,
                dropout=0.3, batch_first=True, activation="gelu")
            self.tr = nn.TransformerEncoder(enc_layer, num_layers=2)
            self.cls_tok = nn.Parameter(torch.zeros(1, 1, d_z))
            nn.init.trunc_normal_(self.cls_tok, std=0.02)
        else:
            raise ValueError(cfg.temporal)

        clf_in = d_z + n_clinical
        if cfg.deeper_clf:
            self.classifier = nn.Sequential(
                nn.Linear(clf_in, 512), nn.GELU(), nn.Dropout(0.5),
                nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(128, 4),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(clf_in, 256), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, 4),
            )

        # loss
        w = torch.ones(4); w[MINF_S2_IDX] = cfg.minf_weight
        self.register_buffer("cls_weight", w)
        if cfg.loss == "focal":
            self.criterion = FocalLoss(gamma=cfg.focal_gamma, weight=w,
                                        label_smoothing=cfg.label_smooth)
        else:
            self.criterion = None   # use F.cross_entropy directly

    def _encode_temporal(self, seq):
        if self.cfg.temporal == "gru":
            out, h_n = self.gru(seq)
            if self.cfg.bidir_gru:
                z = torch.cat([h_n[0], h_n[1]], dim=-1)
                return self.temporal_proj(z)
            return h_n.squeeze(0)
        else:
            B = seq.size(0)
            tok = self.cls_tok.expand(B, -1, -1)
            x = torch.cat([tok, seq], dim=1)
            x = self.pos(x)
            x = self.tr(x)
            return x[:, 0]

    def forward(self, frames, masks, times, labels=None, clinical_feat=None):
        B, N, _, H, W = frames.shape
        device = frames.device
        N_pairs = N - 1

        embs = []
        L_reg = L_bend = L_fold = torch.tensor(0.0, device=device)
        for t in range(N_pairs):
            warped, vel, phi = self.registration.get_warped(
                frames[:, t], frames[:, t + 1])
            _, ncc, bend, fold = self.reg_loss_fn(
                warped, frames[:, t], vel, phi, masks[:, t])
            L_reg = L_reg + ncc; L_bend = L_bend + bend; L_fold = L_fold + fold
            embs.append(self.graph_encoder(masks[:, t], phi))
        L_reg, L_bend, L_fold = (L_reg / N_pairs, L_bend / N_pairs,
                                  L_fold / N_pairs)

        z = self._encode_temporal(torch.stack(embs, dim=1))   # (B, d_z)

        if clinical_feat is not None:
            z_in = torch.cat([z, clinical_feat.to(device)], dim=1)
        else:
            z_in = torch.cat(
                [z, torch.zeros(B, N_CLINICAL_FEATURES, device=device)], dim=1)
        logits = self.classifier(z_in)

        if labels is not None:
            if self.criterion is None:
                cls = F.cross_entropy(logits, labels, weight=self.cls_weight,
                                      label_smoothing=self.cfg.label_smooth)
            else:
                cls = self.criterion(logits, labels)
        else:
            cls = torch.tensor(0.0, device=device)

        total = (cls + self.cfg.alpha * L_reg
                     + self.cfg.beta  * (L_bend + L_fold))
        return Out(logits=logits, total_loss=total, cls_loss=cls,
                   reg_loss=L_reg, bend_loss=L_bend, fold_loss=L_fold)


# ── Train / validate one fold ────────────────────────────────────────────────

def _to_s2(labels):
    return torch.tensor([ORIG_TO_STAGE2[int(l)] for l in labels],
                        device=labels.device)


def train_epoch(model, loader, opt, device):
    model.train()
    tot, n, preds, gts = 0.0, 0, [], []
    for batch in loader:
        f = batch["frames"].to(device); m = batch["masks"].to(device)
        t = batch["times"].to(device);  c = batch["clinical_feat"].to(device)
        y = _to_s2(batch["label"].to(device))
        opt.zero_grad()
        out = model(f, m, t, y, clinical_feat=c)
        out.total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += out.total_loss.item(); n += 1
        preds.extend(out.logits.argmax(1).cpu().tolist())
        gts.extend(y.cpu().tolist())
    return {"loss": tot / max(n, 1),
            "acc":  float((np.array(preds) == np.array(gts)).mean())}


@torch.no_grad()
def val_epoch(model, loader, device, pool="myo"):
    model.eval()
    tot, n = 0.0, 0
    by_pid = defaultdict(list)
    pid_label = {}
    for batch in loader:
        f = batch["frames"].to(device); m = batch["masks"].to(device)
        t = batch["times"].to(device);  c = batch["clinical_feat"].to(device)
        y = _to_s2(batch["label"].to(device))
        out = model(f, m, t, y, clinical_feat=c)
        tot += out.total_loss.item(); n += 1
        logits = out.logits.detach().cpu()
        for i, meta in enumerate(batch["meta"]):
            by_pid[meta["patient_id"]].append(
                (logits[i], float(meta["n_myo_pixels"])))
            pid_label[meta["patient_id"]] = int(y[i].item())

    preds, labels = [], []
    for pid, data in by_pid.items():
        ls = torch.stack([d[0] for d in data])                     # (S, 4)
        if pool == "entropy":
            probs = torch.softmax(ls, dim=1)
            ent   = -(probs * (probs.clamp_min(1e-8)).log()).sum(1)
            w     = torch.softmax(-ent, dim=0)                     # high-conf ↑
        else:  # 'myo'
            w = torch.tensor([d[1] for d in data])
            w = w / (w.sum() + 1e-8)
        mean_l = (ls * w.unsqueeze(1)).sum(0)
        preds.append(int(mean_l.argmax().item()))
        labels.append(pid_label[pid])

    return {"loss": tot / max(n, 1),
            "acc":  float((np.array(preds) == np.array(labels)).mean()),
            "preds": preds, "labels": labels}


def run_fold(cfg, train_pids, val_pids, args, device):
    DS = ElasticAugDataset if cfg.strong_aug else ACDCSliceDataset
    train_ds = DS(args.march9_dir, args.training_dir, train_pids, augment=True)
    val_ds   = ACDCSliceDataset(args.march9_dir, args.training_dir,
                                val_pids, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=2, pin_memory=True)

    if cfg.clinical_only:
        model = ClinicalOnlyModel(N_CLINICAL_FEATURES).to(device)
    else:
        model = ImprovedStage2Model(cfg).to(device)

    # freeze registration for the first args.freeze_reg_epochs (skip for
    # clinical_only — it has no registration module)
    if not cfg.clinical_only:
        for p in model.registration.parameters():
            p.requires_grad = False

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=5e-4)
    sched = make_scheduler(cfg.schedule, opt, args.epochs, cfg.warmup_epochs)

    best_acc = 0.0
    reg_unfrozen = False
    pat = 0
    best_loss = float("inf")

    for ep in range(1, args.epochs + 1):
        if (not cfg.clinical_only and not reg_unfrozen
                and ep > args.freeze_reg_epochs):
            for p in model.registration.parameters():
                p.requires_grad = True
            opt.add_param_group(
                {"params": [p for p in model.registration.parameters()],
                 "lr": args.lr * 0.1})
            reg_unfrozen = True

        tr = train_epoch(model, train_loader, opt, device)
        va = val_epoch(model, val_loader, device, pool=cfg.pool)
        sched.step()

        best_acc = max(best_acc, va["acc"])
        if va["loss"] < best_loss:
            best_loss = va["loss"]; pat = 0
        else:
            pat += 1
        if pat >= args.patience:
            break

    return best_acc, va


# ── K-fold experiment driver ─────────────────────────────────────────────────

def run_experiment(cfg, disease_pids, disease_labels, args, device):
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True,
                          random_state=args.seed)
    fold_accs, all_preds, all_labels = [], [], []
    t0 = time.time()
    for k, (tr_idx, va_idx) in enumerate(
            skf.split(disease_pids, disease_labels)):
        tr_pids = [disease_pids[i] for i in tr_idx]
        va_pids = [disease_pids[i] for i in va_idx]
        _flush(f"  [{cfg.name}] fold {k+1}/{args.n_splits}  "
               f"({len(tr_pids)} tr / {len(va_pids)} va)")
        acc, va = run_fold(cfg, tr_pids, va_pids, args, device)
        fold_accs.append(acc)
        all_preds.extend(va["preds"]); all_labels.extend(va["labels"])
        _flush(f"    best val acc = {acc*100:.1f}%")
    dt = time.time() - t0
    mean = float(np.mean(fold_accs)); std = float(np.std(fold_accs))
    cm   = confusion_matrix(all_labels, all_preds,
                            labels=list(range(4))).tolist()
    per_class = {}
    arr_p = np.array(all_preds); arr_l = np.array(all_labels)
    for i, cls in enumerate(DISEASE_CLASSES):
        mask = arr_l == i
        per_class[cls] = (float((arr_p[mask] == i).mean())
                          if mask.sum() else None)
    return {"name": cfg.name, "mean_acc": mean, "std_acc": std,
            "fold_accs": fold_accs, "per_class": per_class,
            "confusion": cm, "elapsed_s": dt}


# ── Experiment registry ──────────────────────────────────────────────────────

def build_experiments() -> Dict[str, ExpConfig]:
    base = ExpConfig
    return {
        "baseline":      base(name="baseline"),
        "focal_loss":    base(name="focal_loss",    loss="focal"),
        "bidir_gru":     base(name="bidir_gru",     bidir_gru=True),
        "transformer":   base(name="transformer",   temporal="transformer"),
        "warmup_cosine": base(name="warmup_cosine", schedule="warmup_cosine"),
        "strong_aug":    base(name="strong_aug",    strong_aug=True),
        "larger_model":  base(name="larger_model",  d_z=128, deeper_clf=True),
        "conf_pool":     base(name="conf_pool",     pool="entropy"),
        "clinical_only": base(name="clinical_only", clinical_only=True),
        "best_combo":    base(name="best_combo",    loss="focal",
                              bidir_gru=True, schedule="warmup_cosine",
                              pool="entropy"),
    }


# ── CLI / main ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--march9_dir",   required=True)
    p.add_argument("--training_dir", required=True)
    p.add_argument("--n_patients",   type=int, default=100)
    p.add_argument("--epochs",       type=int, default=40)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--patience",     type=int, default=15)
    p.add_argument("--freeze_reg_epochs", type=int, default=10)
    p.add_argument("--n_splits",     type=int, default=3)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 12 patients, 2 folds, 3 epochs.")
    p.add_argument("--experiments", nargs="+", default=None,
                   help="Subset of experiment names; default = all.")
    p.add_argument("--results_file", default="results/improvements_ablation.json")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.n_patients = 12
        args.n_splits   = 2
        args.epochs     = 3
        args.patience   = 5
        args.freeze_reg_epochs = 1
        _flush("[QUICK MODE] 12 patients, 2 folds, 3 epochs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _flush(f"Device: {device}  Seed: {args.seed}")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Disease-only patient list (stratified across DCM/HCM/MINF/RV).
    all_pids = discover_patients_from_images(args.march9_dir)[:args.n_patients]
    pids, labels = [], []
    for pid in all_pids:
        info = parse_patient_info(args.training_dir, pid)
        orig = CLASS_TO_IDX.get(info["group"], 0)
        if orig != NOR_IDX:
            pids.append(pid); labels.append(orig)
    _flush(f"Disease patients: {len(pids)} "
           f"(per-class: " +
           ", ".join(f"{c}={sum(1 for l in labels if l==CLASS_TO_IDX[c])}"
                     for c in DISEASE_CLASSES) + ")")

    experiments = build_experiments()
    chosen = args.experiments or list(experiments.keys())
    unknown = [e for e in chosen if e not in experiments]
    if unknown:
        raise SystemExit(f"Unknown experiments: {unknown}\n"
                         f"Available: {list(experiments.keys())}")

    results = []
    for name in chosen:
        cfg = experiments[name]
        _flush(f"\n{'='*72}\n[Experiment] {name}\n{'='*72}")
        try:
            r = run_experiment(cfg, pids, labels, args, device)
        except Exception as e:
            _flush(f"  !! Failed: {type(e).__name__}: {e}")
            r = {"name": name, "mean_acc": float("nan"),
                 "std_acc": float("nan"), "error": str(e)}
        results.append(r)
        os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
        with open(args.results_file, "w") as f:
            json.dump({"config": vars(args), "results": results}, f, indent=2)

    # Summary table
    _flush("\n" + "=" * 72)
    _flush(f"{'Experiment':<18}{'Mean Acc':>11}{'Std':>9}{'Δ vs base':>12}"
           f"{'Time (s)':>11}")
    _flush("-" * 72)
    base_acc = next((r["mean_acc"] for r in results
                     if r["name"] == "baseline"), None)
    for r in results:
        delta = (f"{(r['mean_acc']-base_acc)*100:+.1f}pt"
                 if base_acc is not None and not math.isnan(r["mean_acc"])
                 else "—")
        _flush(f"{r['name']:<18}"
               f"{r['mean_acc']*100:>10.1f}%"
               f"{r.get('std_acc',0)*100:>8.1f}%"
               f"{delta:>12}"
               f"{r.get('elapsed_s', 0):>11.0f}")
    _flush("=" * 72)
    _flush(f"Results saved → {args.results_file}")


if __name__ == "__main__":
    main()
