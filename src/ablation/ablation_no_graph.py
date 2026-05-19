"""
ablation_no_graph.py — Ablation: remove GAT graph encoder from CardiacMotionODE v3
===================================================================================
Pipeline: Registration → CNN(phi) → GRU → ODE → Classifier  (no GAT)
Replaces the graph attention encoder with a 4-layer CNN + global avg pool on phi.
Compares against full model to quantify contribution of graph-structured encoding.

Usage
-----
  # Full run (~8h on H100, 5-fold):
  cd /home/amo/CINEMRI/src
  CUDA_VISIBLE_DEVICES=5 nohup python -u ablation_no_graph.py \\
      --march9_dir  /home/amo/CINEMRI/data/ACDC/March9Data \\
      --training_dir /home/amo/CINEMRI/data/ACDC/training \\
      --n_patients 100 --epochs 120 --batch_size 8 \\
      --n_splits 5 --patience 30 \\
      --wandb_project cardiac-ode \\
      > logs/abl_no_graph.txt 2>&1 &
  echo "PID: $!"

  # Resume after crash:
  CUDA_VISIBLE_DEVICES=5 nohup python -u ablation_no_graph.py \\
      --march9_dir  /home/amo/CINEMRI/data/ACDC/March9Data \\
      --training_dir /home/amo/CINEMRI/data/ACDC/training \\
      --resume \\
      > logs/abl_no_graph.txt 2>&1 &

  # Check PID:
  cat /home/amo/CINEMRI/src/pid_no_graph.txt

  # Monitor:
  tail -f /home/amo/CINEMRI/src/logs/abl_no_graph.txt

Checkpoints
-----------
  pid_no_graph.txt              — PID of this process
  ckpt_no_graph_fold1.json      — predictions after fold 1 completes
  ...
  results_abl_no_graph.json     — final metrics
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation_base import (
    CardiacMotionNoGraph, run_ablation, make_parser, N_CLINICAL_FEATURES
)

CONDITION = "no_graph"

def model_factory():
    return CardiacMotionNoGraph(
        n_classes=5, d_z=64,
        ode_method='euler',
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
