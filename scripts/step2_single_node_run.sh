#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT

for seed in 1234 2345 3456 4567 5678
do
    deepspeed main.py \
        --data_path local/jsonfile \
        --data_split 0,4,6 \
        --model_name_or_path facebook/opt-350m \
        --num_padding_at_beginning 1 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --max_seq_len 512 \
        --learning_rate 5e-5 \
        --weight_decay 0.1 \
        --num_train_epochs 3 \
        --disable_dropout \
        --gradient_accumulation_steps 1 \
        --lr_scheduler_type cosine \
        --num_warmup_steps 0 \
        --seed $seed \
        --zero_stage $ZERO_STAGE \
        --offload \
        --enable_tensorboard \
        --tensorboard_path step2_tensorboard/$seed \
        --deepspeed \
        --output_dir $OUTPUT \
        &> $OUTPUT/training-$seed.log
done