#!/bin/bash
set -e  # stop on first error

# Create logs directory if not exists
mkdir -p logs

timestamp=$(date +"%Y%m%d_%H%M%S")

# ──────────── FN ────────────
echo "[$timestamp] Running FN - MAGI-X"
python /home/giung/MAGI-TS-3/scripts/claude_magix_exp.py \
  -p /home/giung/MAGI-TS-3/param_configs/data1_fn_magix_noise0.1.config \
  -r 321_results/data1_fn_magix_noise0.1 \
  2>&1 | tee logs/fn_magix_$timestamp.log

echo "[$timestamp] Running FN - NPODE"
python /home/giung/MAGI-TS-3/scripts/claude_magix_exp.py \
  -p /home/giung/MAGI-TS-3/param_configs/data1_fn_npode_noise0.1.config \
  -r 321_results/data1_fn_npode_noise0.1 \
  2>&1 | tee logs/fn_npode_$timestamp.log


# ──────────── LV ────────────
echo "[$timestamp] Running LV - MAGI-X"
python /home/giung/MAGI-TS-3/scripts/claude_magix_exp.py \
  -p /home/giung/MAGI-TS-3/param_configs/data1_lv_magix_noise0.1.config \
  -r 321_results/data1_lv_magix_noise0.1 \
  2>&1 | tee logs/lv_magix_$timestamp.log

echo "[$timestamp] Running LV - NPODE"
python /home/giung/MAGI-TS-3/scripts/claude_magix_exp.py \
  -p /home/giung/MAGI-TS-3/param_configs/data1_lv_npode_noise0.1.config \
  -r 321_results/data1_lv_npode_noise0.1 \
  2>&1 | tee logs/lv_npode_$timestamp.log


# ──────────── HES1 ────────────
echo "[$timestamp] Running HES1 - MAGI-X"
python /home/giung/MAGI-TS-3/scripts/claude_magix_exp.py \
  -p /home/giung/MAGI-TS-3/param_configs/data1_hes1_magix_noise0.1.config \
  -r 321_results/data1_hes1_magix_noise0.1 \
  2>&1 | tee logs/hes1_magix_$timestamp.log

echo "[$timestamp] Running HES1 - NPODE"
python /home/giung/MAGI-TS-3/scripts/claude_magix_exp.py \
  -p /home/giung/MAGI-TS-3/param_configs/data1_hes1_npode_noise0.1.config \
  -r 321_results/data1_hes1_npode_noise0.1 \
  2>&1 | tee logs/hes1_npode_$timestamp.log


# ──────────── NRODE (all systems) ────────────
echo "[$timestamp] Running FN - NRODE"
python /home/giung/MAGI-TS-3/scripts/claude_magix_exp.py \
  -p /home/giung/MAGI-TS-3/param_configs/data1_fn_nrode_noise0.1.config \
  -r 321_results/data1_fn_nrode_noise0.1 \
  2>&1 | tee logs/fn_nrode_$timestamp.log

echo "[$timestamp] Running LV - NRODE"
python /home/giung/MAGI-TS-3/scripts/claude_magix_exp.py \
  -p /home/giung/MAGI-TS-3/param_configs/data1_lv_nrode_noise0.1.config \
  -r 321_results/data1_lv_nrode_noise0.1 \
  2>&1 | tee logs/lv_nrode_$timestamp.log

echo "[$timestamp] Running HES1 - NRODE"
python /home/giung/MAGI-TS-3/scripts/claude_magix_exp.py \
  -p /home/giung/MAGI-TS-3/param_configs/data1_hes1_nrode_noise0.1.config \
  -r 321_results/data1_hes1_nrode_noise0.1 \
  2>&1 | tee logs/hes1_nrode_$timestamp.log

echo "✅ All experiments finished! Logs saved under logs/"
