#!/bin/bash

## Arguments
PRETRAINED_MODEL_NAME=facebook/opt-1.3b
PRETRAINED_MODEL_NAME_SOFT=$(echo "$PRETRAINED_MODEL_NAME" | tr / _)
MASK_FILLING_MODEL_NAME=t5-large

EXTRACT_BATCH_SIZE=384
PERTURB_BATCH_SIZE=160
DETECTGPT_BATCH_SIZE=192

N_GENERATED_SAMPLES=100000
N_PERTURBED_SAMPLES=10
N_SELECTED_SAMPLES=100

## System check.
nvidia-smi
nvidia-smi --query | fgrep "Product Name"

free -h

grep "model name" /proc/cpuinfo | head -1
grep "cpu cores" /proc/cpuinfo | head -1

cat /etc/os-release

## Datetime (KST=UTC+9).
NOWTIME=$(TZ=UTC-9 date "+%Y%m%d-%H%M%S")

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
pip3 install torch transformers easydict tqdm scikit-learn pandas tensorboard
DS_BUILD_OPS=0 pip3 install transformers[deepspeed]
sudo apt-get install -y libaio-dev

ds_report

## Export some cache from home (~/) to block-storage.
## We have only 100GB storage in home directory ;(
export HF_DATASETS_CACHE="/mnt/block-storage/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/mnt/block-storage/.cache/huggingface/transformers"

## Generate.
deepspeed --num_gpus=$NSML_GPU_COUNT extract.py \
    --pretrained_model_name $PRETRAINED_MODEL_NAME \
    --n_generated_samples $N_GENERATED_SAMPLES \
    --batch_size $EXTRACT_BATCH_SIZE \
    --do_sample \
    --min_new_tokens 256 \
    --max_new_tokens 256 \
    --no_repeat_ngram_size 3 \
    --top_p 0.95 \
    --top_k 40 \
    --temperature 1.0 \
    --mi_metrics ce_loss \
    --assets assets \
    --nowtime $NOWTIME \
    --deepspeed ./ds_config/ds_config_zero3.json

## Perturb.
deepspeed --num_gpus=$NSML_GPU_COUNT perturb.py \
    --mask_filling_model_name $MASK_FILLING_MODEL_NAME \
    --pretrained_model_name $PRETRAINED_MODEL_NAME \
    --n_generated_samples $N_GENERATED_SAMPLES \
    --threshold 20 \
    --span_length 2 \
    --buffer_size 2 \
    --pct_words_masked 0.3 \
    --n_perturbed_samples $N_PERTURBED_SAMPLES \
    --batch_size $PERTURB_BATCH_SIZE \
    --do_sample \
    --min_new_tokens 64 \
    --max_new_tokens 256 \
    --no_repeat_ngram_size 3 \
    --top_p 0.95 \
    --top_k 40 \
    --temperature 1.0 \
    --assets assets \
    --nowtime $NOWTIME \
    --deepspeed ./ds_config/ds_config_zero3.json

## DetectGPT
deepspeed --num_gpus=$NSML_GPU_COUNT detectgpt.py \
    --pretrained_model_name $PRETRAINED_MODEL_NAME \
    --n_generated_samples $N_GENERATED_SAMPLES \
    --batch_size $DETECTGPT_BATCH_SIZE \
    --n_perturbed_samples $N_PERTURBED_SAMPLES \
    --test_size 0.2 \
    --assets assets \
    --nowtime $NOWTIME \
    --deepspeed ./ds_config/ds_config_zero3.json

## Install requirements.
git clone https://github.com/microsoft/DeepSpeedExamples.git
cd ./DeepSpeedExamples/applications/DeepSpeed-Chat/
pip install -r requirements.txt

## Copy scripts.
cp /mnt/block-storage/mrt/scripts/step2_single_node_run.sh \
    ./training/step2_reward_model_finetuning/training_scripts/opt/single_node/
cp /mnt/block-storage/mrt/scripts/step2_single_node_run.sh \
    ./training/step3_rlhf_finetuning/training_scripts/opt/single_node/

## Set data.
mkdir ./data
cp /mnt/block-storage/mrt/assets/$PRETRAINED_MODEL_NAME_SOFT/$PRETRAINED_MODEL_NAME_SOFT.$N_GENERATED_SAMPLES.$NOWTIME.perturb.pairs.*.json ./data
mv ./data/*.train.json ./data/train.json
mv ./data/*.eval.json ./data/eval.json

## RLHF step2.
# cd ../step2_reward_model_finetuning/
sudo rm -rf /tmp/data_files/*

cd ./training/step2_reward_model_finetuning/
bash ./training_scripts/opt/single_node/step2_single_node_run.sh 

## RLHF step3.
cd ../step3_rlhf_finetuning/
bash ./training_scripts/opt/single_node/step3_single_node_run.sh \
    $PRETRAINED_MODEL_NAME ../step2_reward_model_finetuning/output 2 2

## Copy outputs.
cd /mnt/block-storage/mrt

mkdir -p ./assets/$PRETRAINED_MODEL_NAME_SOFT/step2
cp -f ./DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/output/training-*.log \
    ./assets/$PRETRAINED_MODEL_NAME_SOFT/step2
cp -rf ./DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/step2_tensorboard/ \
    ./assets/$PRETRAINED_MODEL_NAME_SOFT/step2

mkdir -p ./assets/$PRETRAINED_MODEL_NAME_SOFT/actor_ema
cp -rf ./DeepSpeedExamples/applications/DeepSpeed-Chat/training/step3_rlhf_finetuning/output/actor_ema/ \
    ./assets/$PRETRAINED_MODEL_NAME_SOFT

## Extract on fine-tuned model.
deepspeed --num_gpus=2 extract.py \
    --pretrained_model_name ./assets/$PRETRAINED_MODEL_NAME_SOFT/actor_ema  \
    --n_generated_samples $N_GENERATED_SAMPLES \
    --n_selected_samples 100 \
    --batch_size $DETECTGPT_BATCH_SIZE \
    --do_sample \
    --min_new_tokens 256 \
    --max_new_tokens 256 \
    --no_repeat_ngram_size 3 \
    --top_p 0.95 \
    --top_k 40 \
    --temperature 1.0 \
    --mi_metrics ce_loss ppl zlib lower window \
    --assets assets \
    --do_scoring \
    --nowtime $NOWTIME \
    --deepspeed ./ds_config/ds_config_zero3.json

## Extract.
deepspeed --num_gpus=2 extract.py \
    --load_file \
    --pretrained_model_name $PRETRAINED_MODEL_NAME \
    --n_generated_samples $N_GENERATED_SAMPLES \
    --n_selected_samples 100 \
    --batch_size $DETECTGPT_BATCH_SIZE \
    --do_sample \
    --min_new_tokens 256 \
    --max_new_tokens 256 \
    --no_repeat_ngram_size 3 \
    --top_p 0.95 \
    --top_k 40 \
    --temperature 1.0 \
    --mi_metrics ce_loss ppl zlib lower window \
    --assets assets \
    --do_scoring \
    --nowtime $NOWTIME \
    --deepspeed ./ds_config/ds_config_zero3.json

## Return.
exit 0
