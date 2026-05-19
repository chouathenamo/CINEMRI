"""
evaluate_pipeline.py — Combine Stage 1 + Stage 2 predictions
=============================================================
Takes the saved prediction files from both stages and computes
the final 5-class confusion matrix and accuracy.

Usage:
    python evaluate_pipeline.py \\
        --stage1_preds results/stage1_binary_preds.json \\
        --stage2_preds results/stage2_no_ode_preds.json \\
        --name "Two-stage (No-ODE)"

    python evaluate_pipeline.py \\
        --stage1_preds results/stage1_binary_preds.json \\
        --stage2_preds results/stage2_idea3_preds.json \\
        --name "Two-stage (Idea3)"

Logic:
    For each patient:
      - If stage1 predicts NOR  → final prediction = NOR
      - If stage1 predicts DISEASE → use stage2 prediction (DCM/HCM/MINF/RV)

Note on fold alignment:
    Stage 1 uses all 100 patients with 5-fold CV (stratified, seed=42).
    Stage 2 uses 80 disease patients with 5-fold CV (stratified, seed=42).
    The fold splits are different (different patient pools), so this
    evaluation uses the saved per-patient predictions from each stage
    independently — there is no data leakage since both stages were
    evaluated on held-out patients.

    IMPORTANT: For a fully rigorous evaluation, train and evaluate both
    stages within the same fold (see note at bottom of this file).
"""

import argparse
import json
import numpy as np
from sklearn.metrics import confusion_matrix

# Class name mapping
IDX_TO_CLASS = {0: "NOR", 1: "DCM", 2: "HCM", 3: "MINF", 4: "RV"}
CLASS_TO_IDX = {v: k for k, v in IDX_TO_CLASS.items()}

# Stage 2 label → original 5-class label
DISEASE_CLASSES  = ["DCM", "HCM", "MINF", "RV"]
STAGE2_TO_ORIG   = {i: CLASS_TO_IDX[c] for i, c in enumerate(DISEASE_CLASSES)}


def load_preds(path):
    with open(path) as f:
        return json.load(f)


def combine_predictions(stage1_data, stage2_data):
    """
    Combine stage1 binary predictions with stage2 4-class predictions.

    Returns:
        final_preds  : list of 5-class predictions for all 100 patients
        final_labels : list of 5-class true labels for all 100 patients
        details      : per-patient breakdown
    """
    # Build lookup: pid → stage2 prediction
    stage2_lookup = {}
    for pid, pred_s2, orig in zip(
            stage2_data["pids"],
            stage2_data["preds_s2"],
            stage2_data["orig_labels"]):
        stage2_lookup[pid] = {
            "pred_s2":   pred_s2,
            "orig":      orig,
            "pred_orig": STAGE2_TO_ORIG[pred_s2],
        }

    final_preds  = []
    final_labels = []
    details      = []

    nor_idx = CLASS_TO_IDX["NOR"]

    for pid, pred_b, label_b, orig_label in zip(
            stage1_data["pids"],
            stage1_data["preds_b"],
            stage1_data["labels_b"],
            stage1_data["orig_labels"]):

        true_5class = orig_label

        if pred_b == 0:
            # Stage 1 says NOR → final = NOR
            final_pred = nor_idx
            stage = "S1→NOR"
        else:
            # Stage 1 says DISEASE → use stage 2
            if pid in stage2_lookup:
                final_pred = stage2_lookup[pid]["pred_orig"]
                stage = "S1→DIS, S2"
            else:
                # Patient not in stage2 (shouldn't happen if NOR excluded)
                # Fallback: pick most common disease class
                final_pred = CLASS_TO_IDX["MINF"]
                stage = "S1→DIS, S2_MISSING"

        final_preds.append(final_pred)
        final_labels.append(true_5class)
        details.append({
            "pid":        pid,
            "true":       IDX_TO_CLASS[true_5class],
            "pred":       IDX_TO_CLASS[final_pred],
            "correct":    final_pred == true_5class,
            "stage":      stage,
            "pred_b":     pred_b,
        })

    return final_preds, final_labels, details


def print_results(final_preds, final_labels, details, name):
    preds  = np.array(final_preds)
    labels = np.array(final_labels)
    acc    = (preds == labels).mean()
    cm     = confusion_matrix(labels, preds, labels=[0, 1, 2, 3, 4])

    classes = [IDX_TO_CLASS[i] for i in range(5)]

    print(f"\n{'='*62}")
    print(f"  Pipeline Results — {name}")
    print(f"{'='*62}")
    print(f"  Overall accuracy : {acc*100:.1f}%  "
          f"({(preds==labels).sum()}/{len(labels)})")

    # 4-class accuracy (exclude NOR row and column)
    disease_idx = [1, 2, 3, 4]
    cm_4 = cm[np.ix_(disease_idx, disease_idx)]
    acc4 = cm_4.diagonal().sum() / cm_4.sum() if cm_4.sum() > 0 else 0
    print(f"  4-disease acc    : {acc4*100:.1f}%  (DCM/HCM/MINF/RV)")

    print(f"\n  Per-class:")
    print(f"  {'Class':<8} {'Recall':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'─'*35}")
    for i, cls in enumerate(classes):
        tp      = cm[i, i]
        fn      = cm[i].sum() - tp
        fp      = cm[:, i].sum() - tp
        support = cm[i].sum()
        recall  = tp / support if support > 0 else 0
        print(f"  {cls:<8} {recall*100:>7.1f}%  {tp:>4}  {fp:>4}  {fn:>4}")

    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    header = "       " + "  ".join(f"{c:>5}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {classes[i]:<5}: " + "  ".join(f"{v:>5}" for v in row))

    # Error analysis
    errors = [d for d in details if not d["correct"]]
    print(f"\n  Error breakdown ({len(errors)} errors):")
    s1_errors = [d for d in errors if d["stage"] == "S1→NOR"
                 and d["true"] != "NOR"]
    s1_nor_errors = [d for d in errors if d["stage"] == "S1→DIS, S2"
                     and d["true"] == "NOR"]
    s2_errors = [d for d in errors if d["stage"] == "S1→DIS, S2"
                 and d["true"] != "NOR"]

    print(f"    Stage 1 errors (disease→NOR): {len(s1_errors)}")
    for d in s1_errors:
        print(f"      {d['pid']}  true={d['true']} pred=NOR")

    print(f"    Stage 1 errors (NOR→disease): {len(s1_nor_errors)}")
    for d in s1_nor_errors:
        print(f"      {d['pid']}  true=NOR pred={d['pred']}")

    print(f"    Stage 2 errors (wrong disease): {len(s2_errors)}")

    print(f"{'='*62}\n")

    return {
        "accuracy":         float(acc),
        "acc_4class":       float(acc4),
        "confusion_matrix": cm.tolist(),
        "n_stage1_errors":  len(s1_errors) + len(s1_nor_errors),
        "n_stage2_errors":  len(s2_errors),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_preds", type=str, required=True,
                   help="Path to stage1 _preds.json file")
    p.add_argument("--stage2_preds", type=str, required=True,
                   help="Path to stage2 _preds.json file")
    p.add_argument("--name",         type=str, default="Two-stage pipeline")
    p.add_argument("--output",       type=str, default=None,
                   help="Optional path to save combined results JSON")
    args = p.parse_args()

    print(f"\nLoading stage 1: {args.stage1_preds}")
    stage1_data = load_preds(args.stage1_preds)
    print(f"  {len(stage1_data['pids'])} patients  "
          f"(binary acc={stage1_data['accuracy']*100:.1f}%)")

    print(f"Loading stage 2: {args.stage2_preds}")
    stage2_data = load_preds(args.stage2_preds)
    print(f"  {len(stage2_data['pids'])} patients  "
          f"(4-class acc={stage2_data['accuracy']*100:.1f}%  "
          f"variant={stage2_data.get('variant','?')})")

    final_preds, final_labels, details = combine_predictions(
        stage1_data, stage2_data)

    results = print_results(final_preds, final_labels, details, args.name)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Combined results saved → {args.output}")

    print("\n" + "─"*62)
    print("  COMPARISON SUMMARY")
    print("─"*62)
    print("  Single-stage (no-ODE, 5-class): 75.0%  [reference]")
    print(f"  {args.name}: {results['accuracy']*100:.1f}%")
    delta = results["accuracy"] - 0.75
    print(f"  Delta vs single-stage: {delta*100:+.1f}%")
    print("─"*62 + "\n")

    print("NOTE: This evaluation combines predictions from independently")
    print("trained stage1 and stage2 models. For a fully rigorous")
    print("nested CV evaluation (no information leakage between stages),")
    print("both stages should be trained within the same fold loop.")
    print("The current approach is a reasonable approximation since")
    print("each stage was evaluated on held-out patients.\n")


if __name__ == "__main__":
    main()
