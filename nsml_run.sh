#!/bin/bash

## Get some private information from arguments.
WANDB_API_KEY=$1

## Datetime.
NOWTIME=$(date "+%Y%m%d-%H%M%S")

## Copy the github repository.
rm -rf /mnt/block-storage/*
cp -r /mnt/prj/mrt /mnt/block-storage/mrt
cd /mnt/block-storage/mrt

## Make a symbolic links.
mkdir -p /mnt/prj/$NOWTIME/ckpt
touch /mnt/prj/$NOWTIME/run_log.log

ln -s /mnt/prj/$NOWTIME/ckpt ckpt
ln -s /mnt/prj/$NOWTIME/run_log.log run_log.log

## Install all requirements in local. (i.e., not 'conda env' or 'venv', ...)
sudo apt-get update
sudo apt-get install -y python3-pip

pip3 install --upgrade pip
pip3 install torch transformers lightning easydict black wandb FastAPI
pip3 install transformers[deepspeed]
sudo apt-get install -y libaio-dev

ds_report
wandb login $WANDB_API_KEY

## Export some cache from home (~/) to block-storage.
## We have only 100GB storage in home directory ;(
export HF_DATASETS_CACHE="/mnt/block-storage/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/mnt/block-storage/.cache/huggingface/transformers"

## Record notime to the config.yml file.
echo "nowtime: $NOWTIME" >> config.yml

## Train and record all outputs (stdout, stderr) to a log file.
deepspeed --num_gpus=2 train.py --deepspeed ./assets/ds_config_zero3.json \
  > run_log.log 2>&1

## Return.
exit 0
