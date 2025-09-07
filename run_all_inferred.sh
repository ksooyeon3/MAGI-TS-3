#!/bin/bash
set -e  # stop on first error

# Create logs directory if not exists
mkdir -p logs

timestamp=$(date +"%Y%m%d_%H%M%S")

# ──────────── FN ────────────
echo "[$timestamp] Running FN - MAGI-X"
python /home/giung/MAGI-TS-3/scripts/claude_magix_exp_inferred.py \
  -p /home/giung/MAGI-TS-3/param_configs/data1_fn_magix_noise0.1.config \
  -r 321_results_inferred/data1_fn_magix_noise0.1 \
  2>&1 | tee logs/fn_magix_$timestamp.log


# ──────────── LV ────────────
echo "[$timestamp] Running LV - MAGI-X"
python /home/giung/MAGI-TS-3/scripts/claude_magix_exp_inferred.py \
  -p /home/giung/MAGI-TS-3/param_configs/data1_lv_magix_noise0.1.config \
  -r 321_results_inferred/data1_lv_magix_noise0.1 \
  2>&1 | tee logs/lv_magix_$timestamp.log



# ──────────── HES1 ────────────
echo "[$timestamp] Running HES1 - MAGI-X"
python /home/giung/MAGI-TS-3/scripts/claude_magix_exp_inferred.py \
  -p /home/giung/MAGI-TS-3/param_configs/data1_hes1_magix_noise0.1.config \
  -r 321_results_inferred/data1_hes1_magix_noise0.1 \
  2>&1 | tee logs/hes1_magix_$timestamp.log


echo "✅ All experiments finished! Logs saved under logs/"
