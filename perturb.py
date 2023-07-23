import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import argparse
import logging
import os
import tqdm

import pandas as pd
import numpy as np

from pathlib import Path
from typing import List, Tuple

from src.mask import MaskFillingFunction
from src.utils import (
    define_logger,
    print_config,
    load_results,
    save_results,
    print_rank_0,
)


LOGGER = logging.getLogger(__name__)


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
        default="t5-large",  ## 770M
        help="Name of the model you want to fill mask.",
    )
    ## Model and tokenizer.
    p.add_argument(
        "--pretrained_model_name",
        type=str,
        default="facebook/opt-1.3B",
        help="Name of the model you want to extract.",
    )

    ## Generation.
    p.add_argument(
        "--n_generated_samples",
        type=int,
        default=100_000,
        help="The number of texts you want to sample.",
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
        "--n_perturbed_samples",
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

    ## Nowtime.
    p.add_argument(
        "--nowtime",
        type=str,
        required=True,
        help="The time the learning script was run.",
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

    ## Automated arguments.
    config.save_name = "{}.{}.{}.csv".format(
        config.pretrained_model_name.replace("/", "_"),
        config.n_generated_samples,
        config.nowtime,
    )
    config.save_path = Path(config.assets, config.save_name)

    return config


def get_tokenizer_and_model(
    mask_filling_model_name: str,
) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    ## Load a tokenizer.
    tok = AutoTokenizer.from_pretrained(
        mask_filling_model_name,
        model_max_length=512,
    )

    ## Load a model.
    model = AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name)

    return tok, model


def split_and_mask_texts(
    out: pd.DataFrame,
    threshold: int = 20,
    n_masked_samples: int = 10,
    pct_words_masked: float = 0.3,
    span_length: int = 2,
    buffer_size: int = 2,
    disable_tqdm: bool = False,
):
    ## Define pre-defined arguments.
    calculate_n_spans = lambda x: int(
        np.ceil(pct_words_masked * len(x) / (span_length + buffer_size * 2))
    )
    MASK_STRING = "<<<mask>>>"

    ## Add new columns.
    new_cols = [f"masked_text_{i}" for i in range(n_masked_samples)]
    out.loc[:, new_cols] = np.nan

    ## Split and mask per text.
    dropped_index = []
    desc = f"[+] Split and Masking"
    for i in tqdm.tqdm(out.index, desc=desc, disable=disable_tqdm):
        ## Get item.
        item = out.loc[i]
        masked_texts = {}

        ## Validation check: num words <= 20 ? continue;.
        if len(item.text.split(" ")) <= threshold:
            dropped_index.append(i)
            continue

        ## Generate n masked samples.
        for j in range(n_masked_samples):
            ## Split to tokens.
            tokens = item.text.split(" ")

            n_spans = calculate_n_spans(tokens)
            n_masks = 0

            while n_masks < n_spans:
                ## Select start point and end point randomly.
                sp = np.random.randint(0, len(tokens) - span_length)
                ep = sp + span_length

                search_sp = max(0, sp - buffer_size)
                search_ep = min(len(tokens), ep + buffer_size)

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
            msg = f"[-] num_filled {num_filled} != n_masks {n_masks}"
            assert num_filled == n_masks, msg

            ## Concat tokens to a text.
            masked_text = " ".join(tokens)
            masked_texts[f"masked_text_{j}"] = masked_text

        ## Store masked texts in any order.
        out.loc[i, masked_texts.keys()] = masked_texts.values()

    ## Drop indexes and check validations.
    out = out.drop(dropped_index)
    assert out.isna().sum().sum() == 0
    return out


def predict_masks(
    tok: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: int,
    out: pd.DataFrame,
    batch_size: int,
    do_sample: bool = True,
    min_new_tokens: int = 256,
    max_new_tokens: int = 256,
    no_repeat_ngram_size: int = 3,
    top_p: float = 0.95,
    top_k: int = 40,
    temperature: float = 1.0,
    disable_tqdm: bool = False,
) -> pd.DataFrame:
    ## Predict mask and generate perturb texts function.
    mask_fn = MaskFillingFunction(
        tok,
        model,
        device=device,
        do_sample=do_sample,
        min_new_tokens=min_new_tokens,
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )

    ## Get masked text columns.
    m_cols: List[str] = [c for c in out.columns if c.startswith("masked_text_")]
    p_cols: List[str] = [c.replace("masked", "perturbed") for c in m_cols]

    with tqdm.tqdm(
        total=len(out),
        desc="[+] Predict Masks and Generate Perturbed Texts",
        disable=disable_tqdm,
    ) as pbar:
        ## Calcualate total iterations.
        n_batches = int(np.ceil(len(out) / batch_size))
        for i in range(n_batches):
            ## Get a mini-batch from start to end point.
            sp = i * batch_size
            ep = min((i + 1) * batch_size, len(out))

            for m_col, p_col in zip(m_cols, p_cols):
                ## Set a mini-batch with masked texts.
                batch = out.loc[range(sp, ep), m_col].values.tolist()
                ## |batch|: List[str] = (batch_size,)

                ## Generate perturbed texts.
                perturbed_texts = mask_fn(batch)
                # perturbed_texts = pd.DataFrame(perturbed_texts)

                ## Keep perturbed texts.
                out.loc[range(sp, ep), p_col] = perturbed_texts

            ## Update progress bar.
            pbar.update(ep - sp)

    return out


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
    LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(LOCAL_RANK)
    deepspeed.init_distributed()
    IS_MAIN_PROCESS = LOCAL_RANK <= 0

    ## Load tokenizer and model.
    ## See: https://github.com/huggingface/transformers/issues/3985
    tok, model = get_tokenizer_and_model(config.mask_filling_model_name)
    print_rank_0(
        f"[+] Tokenizer and model are loaded: {config.mask_filling_model_name}",
        LOCAL_RANK,
    )

    ## Initialize deepspeed inference mode.
    ds_engine = deepspeed.init_inference(
        model=model,
        dtype=torch.half,
        tensor_parallel={"tp_size": WORLD_SIZE},
        ## It may cause error in OPT :(
        # replace_with_kernel_inject=True,
    )
    ## Don't forget turn-on evaluation mode.
    ds_engine.module.eval()

    ## ========== LOAD GENERATED TEXTS ==========
    out = load_results(config.save_path)
    print_rank_0(f"[+] Results load from {config.save_path}", LOCAL_RANK)

    ## ========== SPLIT AND MASK TEXTS ==========
    out = split_and_mask_texts(
        out=out,
        threshold=config.threshold,
        n_masked_samples=config.n_perturbed_samples,  ## not masked samples!
        pct_words_masked=config.pct_words_masked,
        span_length=config.span_length,
        buffer_size=config.buffer_size,
        disable_tqdm=not IS_MAIN_PROCESS,
    )
    ## Validation check.
    if len(out) != config.n_generated_samples:
        diff = config.n_generated_samples - len(out)
        msg = f"[!] {diff} samples that do not have enough tokens are dropped."
        print_rank_0(msg, LOCAL_RANK)

    ## ========== PREDICT MASKS AND GENERATE PERTURBED TEXTS  ==========
    out = predict_masks(
        tok=tok,
        model=ds_engine.module,
        device=LOCAL_RANK,
        out=out,
        batch_size=config.batch_size,
        do_sample=config.do_sample,
        min_new_tokens=config.min_new_tokens,
        max_new_tokens=config.max_new_tokens,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
        top_p=config.top_p,
        top_k=config.top_k,
        temperature=config.temperature,
        disable_tqdm=not IS_MAIN_PROCESS,
    )

    ## ========== SAVE TO DATAFRAME ==========
    if IS_MAIN_PROCESS:
        config.save_path = Path(config.save_path).with_suffix(".perturb.json")
        save_results(out, config.save_path)
        print_rank_0(f"[+] Results save to {config.save_path}", LOCAL_RANK)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
