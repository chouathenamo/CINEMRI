"""
ablation_no_clinical.py — Ablation: remove clinical features from CardiacMotionODE v3
======================================================================================
Pipeline: full model but n_clinical=0 (EF, volumes, wall thickness zeroed out).
Compares against full model to quantify contribution of clinical feature fusion.

Usage
-----
  # Full run (~8h on H100, 5-fold):
  cd /home/amo/CINEMRI/src
  CUDA_VISIBLE_DEVICES=5 nohup python -u ablation_no_clinical.py \\
      --march9_dir  /home/amo/CINEMRI/data/ACDC/March9Data \\
      --training_dir /home/amo/CINEMRI/data/ACDC/training \\
      --n_patients 100 --epochs 120 --batch_size 8 \\
      --n_splits 5 --patience 30 \\
      --wandb_project cardiac-ode \\
      > logs/abl_no_clinical.txt 2>&1 &
  echo "PID: $!"

  # Resume after crash:
  CUDA_VISIBLE_DEVICES=5 nohup python -u ablation_no_clinical.py \\
      --march9_dir  /home/amo/CINEMRI/data/ACDC/March9Data \\
      --training_dir /home/amo/CINEMRI/data/ACDC/training \\
      --resume \\
      > logs/abl_no_clinical.txt 2>&1 &

  # Check PID:
  cat /home/amo/CINEMRI/src/pid_no_clinical.txt

  # Monitor:
  tail -f /home/amo/CINEMRI/src/logs/abl_no_clinical.txt

Checkpoints
-----------
  pid_no_clinical.txt              — PID of this process
  ckpt_no_clinical_fold1.json      — predictions after fold 1 completes
  ...
  results_abl_no_clinical.json     — final metrics
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_base import (
    run_ablation, make_parser
)
from model_v3 import CardiacMotionODE

CONDITION = "no_clinical"

def model_factory():
    # Full model but with n_clinical=0 — EF/volume features zeroed out
    return CardiacMotionODE(
        n_classes=5, d_z=64,
        ode_method='euler',
        n_clinical=0,
    )

if __name__ == "__main__":
    import torch
    args   = make_parser(CONDITION).parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}", flush=True)
    print(f"Config : {vars(args)}\n", flush=True)
    os.makedirs("logs", exist_ok=True)
    run_ablation(CONDITION, model_factory, args, device)
