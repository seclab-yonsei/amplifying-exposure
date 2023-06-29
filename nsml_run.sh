#!/bin/bash

## Get some private information from arguments.
WANDB_API_KEY=$1

## Datetime (KST).
ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
NOWTIME=$(date "+%Y%m%d-%H%M%S")

## Copy the github repository.
rm -rf /mnt/block-storage/*
cp -r /mnt/prj/mrt /mnt/block-storage/mrt
cd /mnt/block-storage/mrt

## Make a symbolic links.
mkdir -p /mnt/prj/ckpt/$NOWTIME
touch /mnt/prj/ckpt/$NOWTIME/run_log.log

ln -s /mnt/prj/ckpt/$NOWTIME ./ckpt/$NOWTIME
ln -s /mnt/prj/ckpt/$NOWTIME/run_log.log ./ckpt/$NOWTIME/run_log.log

## Install all requirements in local. (i.e., not 'conda env' or 'venv', ...)
sudo apt-get update
sudo apt-get install -y python3-pip

pip3 install --upgrade pip
pip3 install torch transformers lightning easydict black wandb FastAPI tqdm pandas
DS_BUILD_OPS=0 pip3 install transformers[deepspeed]
sudo apt-get install -y libaio-dev

ds_report
wandb login $WANDB_API_KEY

## Export some cache from home (~/) to block-storage.
## We have only 100GB storage in home directory ;(
export HF_DATASETS_CACHE="/mnt/block-storage/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/mnt/block-storage/.cache/huggingface/transformers"

## Train and record all outputs (stdout, stderr) to a log file.
deepspeed --num_gpus=2 train.py \
  --pretrained_model_name EleutherAI/gpt-neo-1.3B \
  --revision main \
  --samples_per_epoch 10_000 \
  --batch_size 64 \
  --num_workers 24 \
  --wandb_project mrt \
  --ckpt ckpt \
  --every_n_epochs 1 \
  --save_top_k -1 \
  --buffer_size 1_000 \
  --accelerator gpu \
  --devices 2 \
  --precision 16-mixed \
  --accumulate_grad_batches 4 \
  --max_epochs 10 \
  --logging_interval 1 \
  --lr 2e-5 \
  --do_sample \
  --min_new_tokens 64 \
  --max_new_tokens 64 \
  --no_repeat_ngram_size 3 \
  --top_p 0.95 \
  --top_k 40 \
  --rl_n_samples 1 \
  --debug \
  --nowtime $NOWTIME \
  --deepspeed ./assets/ds_config_zero3.json

## Convert checkpoint.
python ./src/zero_to_fp32.py \
  --checkpoint_root_dir ./ckpt/$NOWTIME

## Return.
exit 0
