#!/bin/bash

## Copy the github repository.
rm -rf /mnt/block-storage/*
cp -r /mnt/prj/mrt /mnt/block-storage/mrt
cd /mnt/block-storage/mrt

## Make a symbolic links.
mkdir -p /mnt/prj/assets/
ln -s /mnt/prj/assets/ assets

## Install all requirements in local. (i.e., not 'conda env' or 'venv', ...)
sudo apt-get update
sudo apt-get install -y python3-pip

pip3 install --upgrade pip
pip3 install torch transformers easydict black tqdm pytz scikit-learn

## Export some cache from home (~/) to block-storage.
## We have only 100GB storage in home directory ;(
export HF_DATASETS_CACHE="/mnt/block-storage/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/mnt/block-storage/.cache/huggingface/transformers"

## Train and record all outputs (stdout, stderr) to a log file.
python3 main.py \
    --pretrained_model_name gpt2-xl \
    --revision main \
    --mask_filling_model_name t5-large \
    --mask_filling_model_revision main \
    --device cuda:0 cuda:1 \
    --n 10_000 \
    --batch_size 64 \
    --do_sample \
    --min_new_tokens 256 \
    --max_new_tokens 256 \
    --no_repeat_ngram_size 3 \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 40 \
    --assets assets \
    --span_length 2 \
    --pct_words_masked 0.3 \
    --n_perturb_samples 100 \
    --train_test_split 0.2 \
    --debug

## Return.
exit 0
