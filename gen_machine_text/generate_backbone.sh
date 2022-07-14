#!/bin/bash
module load anaconda/2021a; source activate jack

SAVE_DIR=$1 # e.g. "backbone/gpt2_yelp_l200_boseos_ep5_lr0.0001"
NUM=$2
# TOPK=$3
TOPP=$3

OUTPUT_DIR="gen_n${NUM}_topp"
mkdir -p $OUTPUT_DIR
# FILE_STR="$(basename $SAVE_DIR)"
# OUTPUT_PATH="${OUTPUT_DIR}/${FILE_STR}_topk${TOPK}.txt"
OUTPUT_PATH="${OUTPUT_DIR}/topp${TOPP}.txt"

echo "output_path=${OUTPUT_PATH}"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python generate_backbone.py \
    --backbone $SAVE_DIR \
    --output $OUTPUT_PATH \
    --num $NUM \
    --topp $TOPP
    # --topk $TOPK
    