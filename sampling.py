import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse
import datetime
import json
import logging
import tqdm

import numpy as np

from operator import itemgetter
from pathlib import Path
from pytz import timezone

from src.score import GPTScorer
from src.utils import define_logger, print_config


LOGGER = logging.getLogger(__name__)


def define_argparser() -> argparse.Namespace:
    """Defines the arguments that will be used to execute the code.

    Returns:
        argparse.Namespace: A dictionary whose arguments can be called
    """
    p = argparse.ArgumentParser()

    ## Model and tokenizer.
    p.add_argument(
        "--pretrained_model_name",
        type=str,
        default="EleutherAI/gpt-neo-1.3B",
        help="Name of the model you want to fine-tune.",
    )
    p.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Revision of the pretrained model.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="The device on which the model will be loaded. Multi-gpu inference is not yet implemented.",
    )

    ## Generation.
    p.add_argument(
        "--n",
        type=int,
        default=100_000,
        help="The number of texts you want to sample.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Number of samples to process in one batch.",
    )
    p.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether or not to use sampling; use greedy decoding otherwise.",
    )
    p.add_argument(
        "--min_new_tokens",
        type=int,
        default=64,
        help="The minimum numbers of tokens to generate.",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="The maximum numbers of tokens to generate.",
    )
    p.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help="If set to int > 0, all ngrams of that size can only occur once.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="",
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
    )

    ## Folders.
    p.add_argument(
        "--assets",
        type=str,
        default="assets",
        help="",
    )

    ## Debug.
    p.add_argument(
        "--detect_anomaly",
        action="store_true",
        help="Enable anomaly detection for the autograd engine.",
    )
    p.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Specifies the debugging mode.",
    )

    config = p.parse_args()
    return config


@torch.inference_mode()
def generate(config, tok, model, prompt: str) -> torch.Tensor:
    """One-step to generate text.

    Args:
        config (_type_): A dictionary with configuration items
        tok (_type_): A tokenizer
        model (_type_): A model to generate samples
        prompt (str): Prompt that start a generated sequence

    Returns:
        torch.Tensor: Generated samples in batches
            - |gen_tokens| = (batch_size, length)
    """
    ## Encode it.
    tokens = tok.encode(prompt, return_tensors="pt")
    tokens = tokens.repeat(config.batch_size, 1)
    tokens = tokens.to(device=config.device, non_blocking=True)
    ## |tokens| = (batch_size, 1)

    prompt_len = tokens.size(1)
    assert prompt_len == 1

    ## Generate texts from tokens.
    gen_tokens = model.generate(
        tokens,
        do_sample=config.do_sample,
        min_new_tokens=config.min_new_tokens,
        max_new_tokens=config.max_new_tokens,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
        top_p=config.top_p,
        top_k=config.top_k,
        temperature=config.temperature,
    )
    ## |gen_tokens| = (batch_size, length)

    ## Don't forget detaching from gpu into cpu.
    return gen_tokens.detach().cpu()


@torch.inference_mode()
def inference(tok, model, gen_tokens: torch.Tensor) -> torch.Tensor:
    """Calculate cross entropy loss for the generated tokens without reduction.

    Args:
        model (_type_): A target model
        gen_tokens (torch.Tensor): Generated tokens
            - |gen_tokens| = (batch_size, length)

    Returns:
        torch.Tensor: Cross entropy loss of generated tokens without reduction
            - |ce_loss| = (batch_size,)
    """
    ## Convert batch into logits and the labels.
    labels = gen_tokens.to(model.device)
    logits = model(input_ids=labels, return_dict=True).logits
    ## |labels| = (batch_size, length)
    ## |logits| = (batch_size, length, num_vocabs)

    ## Ignore index.
    labels[labels == tok.pad_token_id] = -100  ## ignore index of CE loss

    ## Calculate ce loss.
    ce_loss = GPTScorer.ce_loss_without_reduction(logits, labels)
    ## |ce_loss| = (batch_size,)
    return ce_loss.detach().cpu()


def save_results(config, rslt: list) -> None:
    """Save the extraction results to a CSV file.

    Args:
        config (_type_): A dictionary with configuration items
        rslt (list): A list where the extraction results are stored as dict
    """
    ## Save the total results.
    kst = timezone("Asia/Seoul")
    nowtime = datetime.datetime.now(kst).strftime("%Y%m%d-%H%M%S")
    model_name = config.pretrained_model_name.replace("/", "_")

    fname = f"{model_name}.{config.n}.{nowtime}.jsonl"  ## not json
    save_path = Path(config.assets, fname)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(rslt, f, indent=4)
        f.write("\n")

    print(f"Results save to {save_path}")


def main(config: argparse.Namespace) -> None:
    ## Print arguments.
    print_config(config)

    ## Set logger.
    define_logger(config.debug)

    ## See: https://sebastianraschka.com/blog/2023/llm-mixed-precision.html
    torch.set_float32_matmul_precision("medium")

    ## Auto-detect error.
    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    ## Load tokenizer and model.
    ## See: https://huggingface.co/EleutherAI/gpt-neo-2.7B
    tok = AutoTokenizer.from_pretrained(
        config.pretrained_model_name,
        revision=config.revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name,
        revision=config.revision,
        pad_token_id=tok.eos_token_id,
        torch_dtype="auto",
    )
    ## Enable padding.
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

    ## Don't forget turn-on evaluation mode.
    _ = model.eval()
    if config.device.startswith("cuda") and torch.cuda.is_available():
        model = model.to(device=config.device, non_blocking=True)

    ## ========== ========== ==========
    ## TEXT SAMPLING
    ## ========== ========== ==========
    rslt = []
    with tqdm.tqdm(total=config.n, desc="Generating Texts") as pbar:
        while True:
            ## Generate sentences with one batch.
            prompt = tok.eos_token
            gen_tokens = generate(config, tok, model, prompt=prompt)

            ## Truncate if we create more than the desired numbers.
            if len(rslt) + len(gen_tokens) > config.n:
                gen_tokens = gen_tokens[: config.n - len(rslt)]

            ## Detokenize and calculate the number of tokens per sample.
            gen_texts = tok.batch_decode(gen_tokens, skip_special_tokens=True)
            n_tokens = (gen_tokens != tok.pad_token_id).sum(dim=1)

            ## Gather it.
            rslt += [
                {"text": g, "n_tokens": int(n)}
                for g, n in zip(gen_texts, n_tokens)
            ]

            ## Update progressbar.
            pbar.update(len(gen_texts))

            if len(rslt) >= config.n:
                break

    ## ========== ========== ==========
    ## MEMBERSHIP INFERENCE
    ## ========== ========== ==========
    rslt = sorted(rslt, key=itemgetter("n_tokens"))
    with tqdm.tqdm(total=len(rslt), desc="Calculating Loss") as pbar:
        n_batches = int(np.ceil(len(rslt) / config.batch_size))

        for i in range(n_batches):
            ## Get a mini-batch from start to end point.
            sp = i * config.batch_size
            ep = min((i + 1) * config.batch_size, len(rslt))

            mini_batch = [rslt[j]["text"] for j in range(sp, ep, 1)]
            mini_batch = tok(mini_batch, padding=True, return_tensors="pt")
            mini_batch = mini_batch.input_ids

            ## Calculate loss.
            ce_loss = inference(tok, model, mini_batch)

            ## Save it.
            for j in range(sp, ep, 1):
                rslt[j]["loss"] = float(ce_loss[j - sp])

            ## Update progress bar.
            pbar.update(len(ce_loss))

    ## Save.
    save_results(config, rslt)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
