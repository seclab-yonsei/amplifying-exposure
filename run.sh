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
    --nowtime 230715-182732 \
    --debug \
    --deepspeed ./ds_config/ds_config_zero3.json

## Perturb.
deepspeed --num_gpus=2 perturb.py \
    --mask_filling_model_name t5-large \
    --threshold 20 \
    --span_length 2 \
    --buffer_size 2 \
    --pct_words_masked 0.3 \
    --n_perturbed_samples 10 \
    --batch_size 320 \
    --do_sample \
    --min_new_tokens 64 \
    --max_new_tokens 256 \
    --no_repeat_ngram_size 3 \
    --top_p 0.95 \
    --top_k 40 \
    --temperature 1.0 \
    --assets assets \
    --nowtime 230715-182732 \
    --debug \
    --deepspeed ./ds_config/ds_config_zero3.json

## DetectGPT
deepspeed --num_gpus=2 detectgpt.py \
    --pretrained_model_name facebook/opt-1.3b \
    --batch_size 128 \
    --n_perturbed_samples 10 \
    --test_size 0.2 \
    --assets assets \
    --nowtime 230715-182732 \
    --debug \
    --deepspeed ./ds_config/ds_config_zero3.json

## RLHF step1
...

## RLHF step2
...

## RLHF step3
...

## Extract.
deepspeed --num_gpus=2 extract.py \
    --pretrained_model_name facebook/opt-1.3b \
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
    --nowtime 230719-164141 \
    --debug \
    --deepspeed ./ds_config/ds_config_zero3.json
