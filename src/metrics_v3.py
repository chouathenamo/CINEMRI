"""
metrics_v3.py — Classification Metrics for CardiacMotionODE
============================================================
Computes per-class Precision, Recall, F1, plus macro/weighted averages.

Usage (standalone — reads your existing results JSON):
    python metrics_v3.py results_full_v3.json

Usage (importable from train_v3 or a notebook):
    from metrics_v3 import compute_all_metrics, print_metrics_table, metrics_to_latex

The function compute_all_metrics() accepts either:
  - (preds, labels) as lists/arrays of integer class indices, OR
  - a confusion matrix as a list-of-lists / np.ndarray (n_classes x n_classes)
"""

import sys
import json
import numpy as np
from typing import Dict, List, Optional, Union

# ── Class names in index order ───────────────────────────────────────────────
IDX_TO_CLASS = {0: "NOR", 1: "DCM", 2: "HCM", 3: "MINF", 4: "RV"}
CLASS_NAMES  = [IDX_TO_CLASS[i] for i in range(5)]


# ── Core computation ──────────────────────────────────────────────────────────

def compute_all_metrics(
    preds_or_cm: Union[List[int], np.ndarray],
    labels:      Optional[List[int]] = None,
    n_classes:   int = 5,
) -> Dict:
    """
    Parameters
    ----------
    preds_or_cm : list of predicted class indices  OR  confusion matrix (n x n)
    labels      : list of true class indices (required if preds_or_cm is a list)
    n_classes   : number of classes

    Returns
    -------
    dict with keys:
        accuracy         float
        macro_f1         float
        weighted_f1      float
        per_class        dict  class_name -> {precision, recall, f1, support}
        confusion_matrix np.ndarray  (rows=true, cols=pred)
    """
    # ── Build confusion matrix ───────────────────────────────────────────────
    if labels is not None:
        # preds_or_cm is a list of predictions
        preds  = np.array(preds_or_cm)
        labels = np.array(labels)
        cm     = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(labels, preds):
            cm[int(t), int(p)] += 1
    else:
        # preds_or_cm IS the confusion matrix
        cm = np.array(preds_or_cm, dtype=int)
        n_classes = cm.shape[0]

    # ── Per-class stats ──────────────────────────────────────────────────────
    per_class = {}
    for c in range(n_classes):
        tp      = cm[c, c]
        fp      = cm[:, c].sum() - tp          # predicted c but not true c
        fn      = cm[c, :].sum() - tp          # true c but not predicted c
        support = cm[c, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        name = IDX_TO_CLASS.get(c, str(c))
        per_class[name] = {
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
            "support":   int(support),
            "tp":        int(tp),
            "fp":        int(fp),
            "fn":        int(fn),
        }

    # ── Aggregate metrics ────────────────────────────────────────────────────
    total_support = sum(v["support"] for v in per_class.values())
    accuracy      = float(cm.diagonal().sum() / cm.sum()) if cm.sum() > 0 else 0.0

    macro_f1      = float(np.mean([v["f1"]      for v in per_class.values()]))
    weighted_f1   = float(sum(
        v["f1"] * v["support"] for v in per_class.values()
    ) / total_support) if total_support > 0 else 0.0

    macro_prec    = float(np.mean([v["precision"] for v in per_class.values()]))
    macro_recall  = float(np.mean([v["recall"]    for v in per_class.values()]))

    # 4-disease accuracy (exclude NOR=0)
    disease_classes = [1, 2, 3, 4]   # DCM, HCM, MINF, RV
    cm_disease      = cm[np.ix_(disease_classes, disease_classes)]
    acc_4class      = float(cm_disease.diagonal().sum() / cm_disease.sum()) \
                      if cm_disease.sum() > 0 else 0.0

    return {
        "accuracy":         accuracy,
        "acc_4class":       acc_4class,
        "macro_f1":         macro_f1,
        "weighted_f1":      weighted_f1,
        "macro_precision":  macro_prec,
        "macro_recall":     macro_recall,
        "per_class":        per_class,
        "confusion_matrix": cm,
    }


# ── Pretty-print table ────────────────────────────────────────────────────────

def print_metrics_table(metrics: Dict, title: str = "Classification Metrics") -> None:
    pc = metrics["per_class"]
    sep = "─" * 62

    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")
    print(f"  Overall accuracy   : {metrics['accuracy']*100:.1f}%")
    print(f"  4-disease accuracy : {metrics['acc_4class']*100:.1f}%  "
          f"(DCM / HCM / MINF / RV only)")
    print(f"  Macro  F1          : {metrics['macro_f1']*100:.1f}%")
    print(f"  Weighted F1        : {metrics['weighted_f1']*100:.1f}%")
    print(f"\n  {sep}")
    print(f"  {'Class':<6}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  "
          f"{'Supp':>5}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    print(f"  {sep}")
    for name, v in pc.items():
        print(f"  {name:<6}  {v['precision']*100:>5.1f}%  "
              f"{v['recall']*100:>5.1f}%  "
              f"{v['f1']*100:>5.1f}%  "
              f"{v['support']:>5}  "
              f"{v['tp']:>4}  {v['fp']:>4}  {v['fn']:>4}")
    print(f"  {sep}")
    print(f"  {'Macro':<6}  {metrics['macro_precision']*100:>5.1f}%  "
          f"{metrics['macro_recall']*100:>5.1f}%  "
          f"{metrics['macro_f1']*100:>5.1f}%")

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion matrix  (rows = true, cols = predicted)")
    print(f"  {'':6} " + "  ".join(f"{c:>5}" for c in CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:<6} " + "  ".join(f"{v:>5}" for v in row))
    print(f"{'═'*62}\n")


# ── LaTeX table row (for paper) ───────────────────────────────────────────────

def metrics_to_latex(
    metrics: Dict,
    model_name: str = "Full model (v3)",
    bold_best: bool = False,
) -> str:
    """
    Returns a single LaTeX table row for copy-paste into your paper.
    Format:
      Model & NOR F1 & DCM F1 & HCM F1 & MINF F1 & RV F1 & Macro F1 & Acc \\\\
    """
    pc   = metrics["per_class"]
    cols = []
    for name in CLASS_NAMES:
        f1 = pc[name]["f1"] * 100
        cols.append(f"{f1:.1f}")

    macro = metrics["macro_f1"]  * 100
    acc   = metrics["accuracy"]  * 100
    acc4  = metrics["acc_4class"] * 100

    row = (f"{model_name} & "
           + " & ".join(cols)
           + f" & {macro:.1f} & {acc:.1f} & {acc4:.1f} \\\\")
    return row


def latex_table_header() -> str:
    return (
        "\\begin{tabular}{lcccccccc}\n"
        "\\toprule\n"
        "Model & NOR F1 & DCM F1 & HCM F1 & MINF F1 & RV F1 "
        "& Macro F1 & Acc (5) & Acc (4) \\\\\n"
        "\\midrule"
    )


# ── Standalone entry point ────────────────────────────────────────────────────

def _load_and_report(json_path: str) -> None:
    with open(json_path) as f:
        data = json.load(f)

    cm = data.get("confusion_matrix")
    if cm is None:
        print(f"ERROR: no 'confusion_matrix' key in {json_path}")
        sys.exit(1)

    metrics = compute_all_metrics(cm)
    run_name = data.get("config", {}).get("wandb_run_name") or json_path
    print_metrics_table(metrics, title=f"Results — {run_name}")

    print("  LaTeX row (paste into Table 1):")
    print("  " + latex_table_header())
    print("  " + metrics_to_latex(metrics, model_name=run_name))
    print("  \\bottomrule\n  \\end{tabular}")

    # Save augmented JSON alongside input
    out_path = json_path.replace(".json", "_with_f1.json")
    export   = {k: v for k, v in data.items()}
    export["f1_metrics"] = {
        "accuracy":        metrics["accuracy"],
        "acc_4class":      metrics["acc_4class"],
        "macro_f1":        metrics["macro_f1"],
        "weighted_f1":     metrics["weighted_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall":    metrics["macro_recall"],
        "per_class": {
            name: {k: v for k, v in vals.items() if k not in ("tp","fp","fn")}
            for name, vals in metrics["per_class"].items()
        },
    }
    # confusion_matrix is np.ndarray — convert back to list for JSON
    export["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"  → Saved augmented results to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python metrics_v3.py <results.json> [results2.json ...]")
        print("\nExample:")
        print("  python metrics_v3.py results_full_v3.json")
        print("  python metrics_v3.py results_*.json          # compare multiple runs")
        sys.exit(0)

    all_metrics = {}
    for path in sys.argv[1:]:
        print(f"\nProcessing: {path}")
        _load_and_report(path)
        with open(path) as f:
            data = json.load(f)
        cm = data.get("confusion_matrix")
        if cm:
            run  = data.get("config", {}).get("wandb_run_name") or path
            all_metrics[run] = compute_all_metrics(cm)

    # If multiple runs: print comparison table
    if len(all_metrics) > 1:
        print("\n" + "═"*62)
        print("  COMPARISON TABLE (LaTeX-ready)")
        print("═"*62)
        print(latex_table_header())
        for name, m in all_metrics.items():
            print(metrics_to_latex(m, model_name=name))
        print("\\bottomrule\n\\end{tabular}")
