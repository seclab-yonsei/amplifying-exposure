## Generate.
deepspeed --num_gpus=2 extract.py \
    --pretrained_model_name facebook/opt-1.3b \
    --n_generated_samples 100_000 \
    --batch_size 256 \
    --do_sample \
    --min_new_tokens 256 \
    --max_new_tokens 256 \
    --no_repeat_ngram_size 3 \
    --top_p 0.95 \
    --top_k 40 \
    --temperature 1.0 \
    --mi_metrics ce_loss \
    --assets assets \
    --nowtime 20230906-173022 \
    --deepspeed ./ds_config/ds_config_zero3.json

## Perturb.
deepspeed --num_gpus=2 perturb.py \
    --mask_filling_model_name t5-large \
    --pretrained_model_name facebook/opt-1.3b \
    --n_generated_samples 100_000 \
    --threshold 20 \
    --span_length 2 \
    --buffer_size 2 \
    --pct_words_masked 0.3 \
    --n_perturbed_samples 10 \
    --batch_size 128 \
    --do_sample \
    --min_new_tokens 64 \
    --max_new_tokens 256 \
    --no_repeat_ngram_size 3 \
    --top_p 0.95 \
    --top_k 40 \
    --temperature 1.0 \
    --assets assets \
    --nowtime 20230811-145623 \
    --deepspeed ./ds_config/ds_config_zero3.json

## DetectGPT
deepspeed --num_gpus=2 detectgpt.py \
    --pretrained_model_name facebook/opt-1.3b \
    --n_generated_samples 100_000 \
    --batch_size 128 \
    --n_perturbed_samples 10 \
    --test_size 0.2 \
    --assets assets \
    --nowtime 20230802-142203 \
    --debug \
    --deepspeed ./ds_config/ds_config_zero3.json

## RLHF step2
rm -rf ~/.cache/huggingface/datasets/*
rm -rf /tmp/data_files/*
rm -rf ./DeepSpeedExamples/applications/DeepSpeed-Chat/data/*
cp ./assets/facebook_opt-1.3b/*.pairs* ./DeepSpeedExamples/applications/DeepSpeed-Chat/data/
mv ./DeepSpeedExamples/applications/DeepSpeed-Chat/data/*train.json ./DeepSpeedExamples/applications/DeepSpeed-Chat/data/train.json
mv ./DeepSpeedExamples/applications/DeepSpeed-Chat/data/*eval.json ./DeepSpeedExamples/applications/DeepSpeed-Chat/data/eval.json

mv ./DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/step2_tensorboard/* ./assets/facebook_opt-350m/step2/step2_tensorboard/
mv ./DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/output/*.log ./assets/facebook_opt-350m/step2/

tensorboard --logdir=./assets/facebook_opt-350m/step2/step2_tensorboard
tensorboard --logdir=./DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/step2_tensorboard/

## RLHF step3
...

## Extract on fine-tuned model.
deepspeed --num_gpus=2 extract.py \
    --pretrained_model_name ./assets/facebook_opt-1.3b/actor_ema \
    --n_generated_samples 100_000 \
    --n_selected_samples 100 \
    --batch_size 128 \
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
    --nowtime 20230909-225923 \
    --deepspeed ./ds_config/ds_config_zero3.json

## Extract.
deepspeed --num_gpus=2 extract.py \
    --load_file \
    --pretrained_model_name facebook/opt-350m \
    --n_generated_samples 100_000 \
    --n_selected_samples 100 \
    --batch_size 128 \
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
    --nowtime 20230906-173022 \
    --deepspeed ./ds_config/ds_config_zero3.json
