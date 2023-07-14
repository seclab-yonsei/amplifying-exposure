import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import argparse
import logging
import os
import tqdm

import numpy as np

from pathlib import Path
from typing import List, Final

from src.mask import MaskFillingFunction
from src.utils import define_logger, print_config, load_results, save_results


LOGGER = logging.getLogger(__name__)

MASK_STRING: Final[str] = "<<<mask>>>"


def define_argparser() -> argparse.Namespace:
    """Defines the arguments that will be used to execute the code.

    Returns:
        argparse.Namespace: A dictionary whose arguments can be called
    """
    p = argparse.ArgumentParser()

    ## Model and tokenizer.
    p.add_argument(
        "--mask_filling_model_name",
        type=str,
        default="t5-3b",  ## 770M
        help="Name of the model you want to fill mask.",
    )

    ## DetectGPT.
    p.add_argument(
        "--threshold",
        type=int,
        default=20,
        help="",
    )
    p.add_argument(
        "--span_length",
        type=int,
        default=2,
        help="Number of consecutive words to mask.",
    )
    p.add_argument(
        "--buffer_size",
        type=int,
        default=2,
        help="",
    )
    p.add_argument(
        ## Pct masked is actually calculated as:
        ##  - pct_words_masked * (span_length / (span_length + 2 * buffer_size))
        "--pct_words_masked",
        type=float,
        default=0.3,
        help="Percentage of words to be masked.",
    )
    p.add_argument(
        "--n_perturb_samples",
        type=int,
        default=3,
        help="Number of samples to perturb in one sample.",
    )

    ## Generation.
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
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
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The value used to modulate the next token probabilities.",
    )

    ## Folders.
    p.add_argument(
        "--assets",
        type=str,
        default="assets",
        help="The directory where the experiment results will be stored.",
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

    ## Deepspeed arguments (meaningless).
    p.add_argument(
        "--local_rank",
        type=int,
        default=None,
    )
    p.add_argument(
        "--deepspeed",
        type=str,
        default=None,
    )

    config = p.parse_args()
    return config


def calculate_n_spans(
    tokens: List[str],
    pct_words_masked: float,
    span_length: int,
    buffer_size: int,
) -> int:
    n_spans = pct_words_masked * len(tokens) / (span_length + buffer_size * 2)
    n_spans = int(np.ceil(n_spans))
    return n_spans


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

    ## Distributed setup.
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()

    ## Load tokenizer and model (T5).
    ## See: https://github.com/huggingface/transformers/issues/3985
    tok = AutoTokenizer.from_pretrained(
        config.mask_filling_model_name,
        model_max_length=512,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.mask_filling_model_name,
    )

    ## Initialize deepspeed inference mode.
    ds_engine = deepspeed.init_inference(
        model=model,
        dtype=torch.half,
        tensor_parallel={"tp_size": world_size},
        ## It may cause error in OPT :(
        # replace_with_kernel_inject=True,
    )
    ## Don't forget turn-on evaluation mode.
    ds_engine.module.eval()

    ## Load results.
    rslt_, save_path = load_results(assets=config.assets)

    ## Results.
    rslt = []

    ## Split and mask per text.
    desc = f"🚀 Split and Masking"
    for item in tqdm.tqdm(rslt_, desc=desc, position=local_rank):
        ## Unpack.
        text = item["text"]
        perturbed_texts = []

        ## Validation check: num words <= 20 ? continue;.
        if len(text.split(" ")) <= config.threshold:
            continue

        ## Generate n perturb samples.
        for i in range(config.n_perturb_samples):
            ## Split to tokens.
            tokens = text.split(" ")

            n_spans = calculate_n_spans(tokens)
            n_masks = 0

            while n_masks < n_spans:
                sp = np.random.randint(0, len(tokens) - config.span_length)
                ep = sp + config.span_length

                search_sp = max(0, sp - config.buffer_size)
                search_ep = min(len(tokens), ep + config.buffer_size)

                ## If mask not in tokens, then mask tokens.
                if MASK_STRING not in tokens[search_sp:search_ep]:
                    tokens[sp:ep] = [MASK_STRING]
                    n_masks += 1

            ## Replace each occurrence of MASK_STRING with <extra_id_NUM>,
            ## where NUM increments
            num_filled = 0
            for idx, token in enumerate(tokens):
                if token == MASK_STRING:
                    tokens[idx] = f"<extra_id_{num_filled}>"
                    num_filled += 1

            ## Validation check: all masks replaced to t5-mask tokens?
            msg = f"num_filled {num_filled} != n_masks {n_masks}"
            assert num_filled == n_masks, msg

            ## Concat tokens to a text.
            perturbed_text = " ".join(tokens)
            perturbed_texts.append({"id": i, "masked_text": perturbed_text})

        item["perturbed_texts"] = perturbed_texts
        rslt.append(item)

    ## Check.
    diff = len(rslt_) - len(rslt)
    msg = f"🙄 {diff} sample(s) that were not properly sampled were dropped."
    if diff > 0:
        print(msg)

    ## Predict mask and generate perturb texts.
    mask_fn = MaskFillingFunction(
        tok,
        ds_engine.module,
        device=local_rank,
        do_sample=config.do_sample,
        min_new_tokens=config.min_new_tokens,
        max_new_tokens=config.max_new_tokens,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
        top_p=config.top_p,
        top_k=config.top_k,
        temperature=config.temperature,
    )

    desc = "🐢 Predict Masks and Generate Perturbed Texts"
    with tqdm.tqdm(total=len(rslt), desc=desc, position=local_rank) as pbar:
        n_batches = int(np.ceil(len(rslt) / config.batch_size))

        for i in range(n_batches):
            ## Get a mini-batch from start to end point.
            sp = i * config.batch_size
            ep = min((i + 1) * config.batch_size, len(rslt))

            for j in range(config.n_perturb_samples):
                ## Gather masked texts.
                masked_texts = [
                    rslt[k]["perturbed_texts"][j]["masked_text"]
                    for k in range(sp, ep, 1)
                ]

                ## Generate perturbed texts.
                perturbed_texts = mask_fn(masked_texts)

                ## Keep perturbed texts.
                for idx, k in enumerate(range(sp, ep, 1)):
                    rslt[k]["perturbed_texts"][j][
                        "perturbed_text"
                    ] = perturbed_texts[idx]

            ## Update progress bar.
            pbar.update(ep - sp)

    ## Save.
    save_results(rslt, save_path)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
