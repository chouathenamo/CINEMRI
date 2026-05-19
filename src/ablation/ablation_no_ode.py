"""
ablation_no_ode.py — Ablation: remove Neural ODE from CardiacMotionODE v3
=========================================================================
Pipeline: Registration → Graph(GAT) → GRU → Classifier  (no ODE integration)
Compares against full model to quantify contribution of continuous dynamics.

Usage
-----
  # Full run (~8h on H100, 5-fold):
  cd /home/amo/CINEMRI/src
  CUDA_VISIBLE_DEVICES=5 nohup python -u ablation_no_ode.py \\
      --march9_dir  /home/amo/CINEMRI/data/ACDC/March9Data \\
      --training_dir /home/amo/CINEMRI/data/ACDC/training \\
      --n_patients 100 --epochs 120 --batch_size 8 \\
      --n_splits 5 --patience 30 \\
      --wandb_project cardiac-ode \\
      > logs/abl_no_ode.txt 2>&1 &
  echo "PID: $!"

  # Resume after crash (auto-loads fold checkpoints):
  CUDA_VISIBLE_DEVICES=5 nohup python -u ablation_no_ode.py \\
      --march9_dir  /home/amo/CINEMRI/data/ACDC/March9Data \\
      --training_dir /home/amo/CINEMRI/data/ACDC/training \\
      --resume \\
      > logs/abl_no_ode.txt 2>&1 &

  # Check PID (written to pid_no_ode.txt at startup):
  cat /home/amo/CINEMRI/src/pid_no_ode.txt

  # Monitor:
  tail -f /home/amo/CINEMRI/src/logs/abl_no_ode.txt

Checkpoints
-----------
  pid_no_ode.txt              — PID of this process (safe to verify)
  ckpt_no_ode_fold1.json      — predictions after fold 1 completes
  ckpt_no_ode_fold2.json      — predictions after fold 2 completes
  ...
  results_abl_no_ode.json     — final metrics (written at the end)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_base import (
    CardiacMotionNoODE, run_ablation, make_parser, N_CLINICAL_FEATURES
)

CONDITION = "no_ode"

def model_factory():
    return CardiacMotionNoODE(
        n_classes=5, d_z=64,
        n_clinical=N_CLINICAL_FEATURES,
    )

if __name__ == "__main__":
    import torch
    args   = make_parser(CONDITION).parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}", flush=True)
    print(f"Config : {vars(args)}\n", flush=True)
    os.makedirs("logs", exist_ok=True)
    run_ablation(CONDITION, model_factory, args, device)
