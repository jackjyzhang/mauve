#!/bin/bash
module load anaconda/2021a; source activate jack

SAVE_DIR=$1 # e.g. "backbone/gpt2_yelp_l200_boseos_ep5_lr0.0001"
IS_FT=$2
NUM=$3
HPARAM_NAME=$4 # topk, topp, temp
HPARAM_VALUE=$5 # topk: positive int, topp: 0 to 1, temp: >=0
DIR_PREFIX=$6 # can be empty, should contain model name / dataset (e.g. gpt2small, yelp, etc)

OUTPUT_DIR="gen_n${NUM}/${DIR_PREFIX}_${HPARAM_NAME}"
mkdir -p $OUTPUT_DIR
OUTPUT_PATH="${OUTPUT_DIR}/${HPARAM_NAME}${HPARAM_VALUE}.txt"

if [[ $IS_FT -eq 1 ]]; then
    IS_FT_FLAG="--is_finetuned"
fi

echo "output_path=${OUTPUT_PATH}"
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python generate_backbone.py \
    --backbone $SAVE_DIR \
    --output $OUTPUT_PATH \
    --num $NUM $IS_FT_FLAG \
    --$HPARAM_NAME $HPARAM_VALUE
    