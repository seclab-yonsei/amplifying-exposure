## Sampling.
python main.py \
    --pretrained_model_name facebook/opt-1.3b \
    --mask_filling_model_name t5-large \
    --device cuda:0 cuda:1 \
    --n 10_000 \
    --batch_size 32 \
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
    --n_perturb_samples 10 \
    --test_size 0.2 \
    --debug
    