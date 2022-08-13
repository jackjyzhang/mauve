#!/bin/bash
module load anaconda/2021a; source activate jack

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python eval_mauve.py $@
