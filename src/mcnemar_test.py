"""
mcnemar_test.py — McNemar's test for pairwise model comparison
==============================================================
Usage:
    python mcnemar_test.py \
        --results_a results_abl_no_ode_with_f1.json \
        --results_b results_full_v3_p30_with_f1.json \
        --name_a "No-ODE" \
        --name_b "Full model p30"

Requirements:
    pip install scipy statsmodels --break-system-packages

IMPORTANT — this script works with confusion matrices (existing results).
For a fully paired test you need per-patient predictions (see save_preds
flag in train scripts). The confusion-matrix-based approximation is still
informative and commonly used in medical imaging papers.

If you have per-patient predictions saved (new runs with --save_preds),
pass --preds_a and --preds_b instead of --results_a/b.
"""

import argparse
import json
import numpy as np
from scipy.stats import binomtest


# ── McNemar from confusion matrices (approximate) ────────────────────────────

def mcnemar_from_cm(cm_a, cm_b, n_classes=5):
    """
    Approximate McNemar using per-class TP counts from confusion matrices.
    Both models evaluated on the same 100 patients (same folds, seed=42).

    correct_a[i] = 1 if model A correctly predicted patient i
    correct_b[i] = 1 if model B correctly predicted patient i

    We reconstruct per-patient correctness from the diagonal of the
    confusion matrix. This is an approximation because we don't know
    which specific patients were correct — we only know counts per class.

    For a fully exact test, use --preds_a / --preds_b.
    """
    cm_a = np.array(cm_a)
    cm_b = np.array(cm_b)

    n_total = cm_a.sum()
    correct_a = cm_a.diagonal().sum()
    correct_b = cm_b.diagonal().sum()

    # Per-class analysis
    print(f"\n{'─'*60}")
    print(f"  Per-class TP comparison:")
    print(f"  {'Class':<8} {'Model A':>8} {'Model B':>8} {'Diff':>8}")
    print(f"{'─'*60}")
    classes = ['NOR', 'DCM', 'HCM', 'MINF', 'RV']
    for i, cls in enumerate(classes):
        tp_a = cm_a[i, i]
        tp_b = cm_b[i, i]
        print(f"  {cls:<8} {tp_a:>8} {tp_b:>8} {tp_b-tp_a:>+8}")
    print(f"{'─'*60}")
    print(f"  {'Total':<8} {correct_a:>8} {correct_b:>8} {correct_b-correct_a:>+8}")
    print(f"  {'Accuracy':<8} {correct_a/n_total*100:>7.1f}% {correct_b/n_total*100:>7.1f}%")

    return correct_a, correct_b, n_total


def mcnemar_from_preds(preds_a, preds_b, labels):
    """
    Exact McNemar's test using paired per-patient predictions.
    preds_a, preds_b, labels: lists of length 100 (all patients).
    """
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    labels  = np.array(labels)

    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    # b: A correct, B wrong
    # c: A wrong,   B correct
    b = ((correct_a == True)  & (correct_b == False)).sum()
    c = ((correct_a == False) & (correct_b == True)).sum()

    return b, c


def run_mcnemar(b, c, name_a, name_b):
    """
    McNemar's test: H0 = both models have same error rate.
    Uses exact binomial test (better than chi-squared for small n).
    """
    n_discordant = b + c

    print(f"\n{'='*60}")
    print(f"  McNemar's Test: {name_a}  vs  {name_b}")
    print(f"{'='*60}")
    print(f"  Discordant pairs:")
    print(f"    {name_a} correct, {name_b} wrong : b = {b}")
    print(f"    {name_a} wrong,   {name_b} correct: c = {c}")
    print(f"    Total discordant (b+c)        : {n_discordant}")

    if n_discordant == 0:
        print(f"\n  Models are identical — no discordant pairs.")
        return

    # Exact binomial test (two-tailed)
    p_value = binomtest(b, n_discordant, 0.5).pvalue

    print(f"\n  Exact binomial p-value : {p_value:.4f}")

    if p_value < 0.001:
        significance = "*** (p < 0.001, highly significant)"
    elif p_value < 0.01:
        significance = "** (p < 0.01, very significant)"
    elif p_value < 0.05:
        significance = "* (p < 0.05, significant)"
    elif p_value < 0.10:
        significance = "† (p < 0.10, marginal)"
    else:
        significance = "n.s. (not significant)"

    print(f"  Significance           : {significance}")

    winner = name_a if b > c else name_b
    if p_value < 0.05:
        print(f"\n  → {winner} is significantly better (p={p_value:.4f})")
    else:
        print(f"\n  → No significant difference between models (p={p_value:.4f})")
        print(f"     With only 100 patients, you need ~10+ discordant pairs")
        print(f"     favouring one model to reach significance.")

    print(f"{'='*60}\n")
    return p_value


def load_results(path):
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    # Confusion-matrix mode (existing results)
    p.add_argument("--results_a",  type=str, default=None,
                   help="Path to results JSON for model A")
    p.add_argument("--results_b",  type=str, default=None,
                   help="Path to results JSON for model B")
    # Paired predictions mode (new runs with save_preds)
    p.add_argument("--preds_a",    type=str, default=None,
                   help="Path to predictions JSON for model A")
    p.add_argument("--preds_b",    type=str, default=None,
                   help="Path to predictions JSON for model B")
    p.add_argument("--name_a",     type=str, default="Model A")
    p.add_argument("--name_b",     type=str, default="Model B")
    args = p.parse_args()

    if args.preds_a and args.preds_b:
        # ── Exact McNemar from paired predictions ─────────────────────────
        print(f"\nMode: EXACT (paired per-patient predictions)")
        with open(args.preds_a) as f: data_a = json.load(f)
        with open(args.preds_b) as f: data_b = json.load(f)

        preds_a = data_a["preds"]
        preds_b = data_b["preds"]
        labels  = data_a["labels"]   # same labels for both (same fold splits)

        assert len(preds_a) == len(preds_b) == len(labels), \
            "Prediction arrays must all be length 100"

        b, c = mcnemar_from_preds(preds_a, preds_b, labels)
        run_mcnemar(b, c, args.name_a, args.name_b)

    elif args.results_a and args.results_b:
        # ── Approximate McNemar from confusion matrices ───────────────────
        print(f"\nMode: APPROXIMATE (confusion matrix based)")
        print(f"Note: for exact results, use --preds_a / --preds_b")
        print(f"      (requires re-running with save_preds=True)\n")

        res_a = load_results(args.results_a)
        res_b = load_results(args.results_b)

        # Get accuracy
        acc_a = res_a.get("accuracy") or res_a.get("f1_metrics", {}).get("accuracy")
        acc_b = res_b.get("accuracy") or res_b.get("f1_metrics", {}).get("accuracy")
        cm_a  = res_a["confusion_matrix"]
        cm_b  = res_b["confusion_matrix"]

        print(f"  {args.name_a:<25} accuracy: {acc_a*100:.1f}%")
        print(f"  {args.name_b:<25} accuracy: {acc_b*100:.1f}%")

        correct_a, correct_b, n_total = mcnemar_from_cm(cm_a, cm_b)

        # For approximate test: assume worst case discordance
        # (all errors are discordant between models)
        errors_a = n_total - correct_a
        errors_b = n_total - correct_b

        # Conservative approximation:
        # b ≈ errors_b (cases B got wrong that A might have right)
        # c ≈ errors_a (cases A got wrong that B might have right)
        # This overstates discordance — real test needs paired data
        b_approx = int(errors_b)
        c_approx = int(errors_a)

        print(f"\n  ⚠️  APPROXIMATE ONLY — treating all errors as discordant.")
        print(f"     This is a conservative upper bound on significance.")
        print(f"     Re-run with paired predictions for exact result.\n")

        run_mcnemar(b_approx, c_approx, args.name_a, args.name_b)

    else:
        print("Provide either --results_a + --results_b  or  --preds_a + --preds_b")
        print("\nExample (approximate, existing results):")
        print("  python mcnemar_test.py \\")
        print("    --results_a results_abl_no_ode_with_f1.json \\")
        print("    --results_b results_full_v3_p30_with_f1.json \\")
        print("    --name_a 'No-ODE' --name_b 'Full model p30'")
        print("\nExample (exact, after re-running with save_preds):")
        print("  python mcnemar_test.py \\")
        print("    --preds_a preds_no_ode.json \\")
        print("    --preds_b preds_idea3.json \\")
        print("    --name_a 'No-ODE' --name_b 'Idea3'")


if __name__ == "__main__":
    main()
