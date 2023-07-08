## Sampling.
python main.py \
    --pretrained_model_name EleutherAI/gpt-neo-1.3B \
    --revision main \
    --mask_filling_model_name t5-large \
    --mask_filling_model_revision main \
    --device cuda:0 cuda:1 \
    --n 100 \
    --batch_size 24 \
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
    --n_perturb_samples 3 \
    --debug
    