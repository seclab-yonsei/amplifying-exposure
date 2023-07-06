## Sampling.
python sampling_texts.py \
    --pretrained_model_name EleutherAI/gpt-neo-1.3B \
    --revision main \
    --device cuda:0 \
    --n 1000 \
    --batch_size 24 \
    --do_sample \
    --min_new_tokens 64 \
    --max_new_tokens 256 \
    --no_repeat_ngram_size 3 \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 40 \
    --assets assets \
    --debug

## Calculate loss.
python calculate_loss.py \
    --pretrained_model_name EleutherAI/gpt-neo-1.3B \
    --revision main \
    --device cuda:0 \
    --save_file \
    --batch_size 64 \
    --assets assets \
    --debug
