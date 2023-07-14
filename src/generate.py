import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.inference_mode()
def generate(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: int,
    batch_size: int,
    prompt: str = "",
    do_sample: bool = True,
    min_new_tokens: int = 256,
    max_new_tokens: int = 256,
    no_repeat_ngram_size: int = 3,
    top_p: float = 0.95,
    top_k: int = 40,
    temperature: float = 1.0,
) -> torch.Tensor:
    """One-step to generate text.

    Args:
        tok (AutoTokenizer): Tokenizer function
        model (AutoModelForCausalLM): Causal LM to generate text
        batch_size (int): Number of samples to process in one batch
        prompt (str, optional): Input prompt tokens. Defaults to "".
        do_sample (bool, optional): Whether or not to use sampling; use greedy decoding otherwise. Defaults to True.
        min_new_tokens (int, optional): The minimum numbers of tokens to generate. Defaults to 256.
        max_new_tokens (int, optional): The maximum numbers of tokens to generate. Defaults to 256.
        no_repeat_ngram_size (int, optional): If set to int > 0, all ngrams of that size can only occur once. Defaults to 3.
        top_p (float, optional): Top-p sampling coefficient. Defaults to 0.95.
        top_k (int, optional): Top-k sampling coefficient. Defaults to 40.
        temperature (float, optional): The value used to modulate the next token probabilities. Defaults to 1.0.

    Returns:
        torch.Tensor: Generated samples in a batch
            - |gen_tokens| = (batch_size, 1 + length)
    """
    ## Prepare prompts.
    prompts = tok.encode(prompt, return_tensors="pt", add_special_tokens=True)
    ## |prompts| = (1,)

    ## Make a batch and move it to model's device.
    prompts = prompts.repeat(batch_size, 1)
    prompts = prompts.to(device=device)
    ## |prompts| = (batch_size, 1)

    ## Prompts must have only one token.
    prompt_len = prompts.size(1)
    assert prompt_len == 1, prompt_len

    ## Generate texts from tokens.
    ## See: https://huggingface.co/docs/transformers/main_classes/deepspeed#custom-deepspeed-zero-inference
    tokens = model.generate(
        prompts,
        do_sample=do_sample,
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        synced_gpus=True,
    )
    ## |tokens| = (batch_size, 1 + length)

    ## Don't forget detaching from gpu into cpu.
    tokens = tokens.detach().cpu()

    return tokens
