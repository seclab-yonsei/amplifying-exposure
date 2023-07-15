import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse
import logging
import os
import tqdm

import numpy as np
import pandas as pd

from src.generate import generate
from src.score import ScoreFunction
from src.utils import (
    define_logger,
    print_config,
    calculate_similarity,
    save_results,
)

## To avoid warnings about parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

    ## Generation.
    p.add_argument(
        "--n_generated_samples",
        type=int,
        default=100_000,
        help="The number of texts you want to sample.",
    )
    p.add_argument(
        "--n_selected_samples",
        type=int,
        default=100,
        help="The number of texts you want to screen out of the sampled texts.",
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

    ## Membership inference.
    p.add_argument(
        "--mi_metrics",
        nargs="+",
        default=["ce_loss", "ppl", "zlib", "lower", "window"],
        help="Membership inference metrics we want to use.",
    )

    ## Folders.
    p.add_argument(
        "--assets",
        type=str,
        default="assets",
        help="The directory where the experiment results will be stored.",
    )

    ## Scoring.
    p.add_argument(
        "--do_scoring",
        action="store_true",
        help="Whether to proceed with membership inference.",
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
    return config


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

    ## Load tokenizer and model.
    tok = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name,
        pad_token_id=tok.eos_token_id,
    )
    ## Enable padding.
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

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

    ## Text sampling.
    rslt = []
    with tqdm.tqdm(
        total=config.n_generated_samples,
        desc="🛴 Generating Texts",
    ) as pbar:
        while True:
            ## Generate sentences with one batch.
            prompt = ""
            tokens = generate(
                tok,
                ds_engine.module,
                device=local_rank,
                batch_size=config.batch_size,
                prompt=prompt,
                do_sample=config.do_sample,
                min_new_tokens=config.min_new_tokens,
                max_new_tokens=config.max_new_tokens,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                top_p=config.top_p,
                top_k=config.top_k,
                temperature=config.temperature,
            )

            ## Truncate if we create more than the desired numbers.
            if len(rslt) + len(tokens) > config.n_generated_samples:
                tokens = tokens[: config.n_generated_samples - len(rslt)]

            ## Detokenize and calculate the number of tokens per sample.
            texts = tok.batch_decode(tokens, skip_special_tokens=True)
            n_tokens = (tokens != tok.pad_token_id).sum(dim=1)

            ## Gather it.
            rslt += [
                {"text": t, "n_tokens": int(n), "n_words": len(t.split(" "))}
                for t, n in zip(texts, n_tokens)
            ]

            ## Update progressbar.
            pbar.update(len(texts))

            if len(rslt) >= config.n_generated_samples:
                break

    ## Membership inference.
    score_fn = ScoreFunction(
        tok,
        ds_engine.module,
        device=local_rank,
        mi_metrics=config.mi_metrics,
    )

    with tqdm.tqdm(total=len(rslt), desc="🚂 Calculating Loss") as pbar:
        n_batches = int(np.ceil(len(rslt) / config.batch_size))

        for i in range(n_batches):
            ## Get a mini-batch from start to end point.
            sp = i * config.batch_size
            ep = min((i + 1) * config.batch_size, len(rslt))
            batch = [rslt[j]["text"] for j in range(sp, ep, 1)]

            ## Get membership inference metrics.
            rslt_ = score_fn(batch)

            ## Save the results.
            for j, k in enumerate(range(sp, ep, 1)):
                for key, value in rslt_.items():
                    rslt[k][key] = float(value[j])

            ## Update progress bar.
            pbar.update(len(batch))

    ## If we only want to generate and calculate loss...
    if not config.do_scoring:
        if local_rank == 0:
            ## Save.
            save_name = ".".join(
                [
                    config.pretrained_model_name.replace("/", "_"),
                    str(config.n_generated_samples),
                    config.nowtime,
                    "json",
                ]
            )
            save_results(rslt, save_name, config.assets)

        ## And exit the code.
        return

    ## Scoring.
    ##  - Score 1: only perplexity (lower best -> multiply -1 to higher best)
    ##  - Score 2: zlib entropy / perpledixy (higher best)
    ##  - Score 3: lowercase perplexity ratio (higher best)
    ##  - Score 4: window perpledixy (lower best -> multiply -1 to higher best)
    df = pd.DataFrame(rslt)
    df.loc[:, "score1"] = -np.log(df.loc[:, "ppl"])
    df.loc[:, "score2"] = df.loc[:, "zlib"] / np.log(df.loc[:, "ppl"])
    df.loc[:, "score3"] = np.log(df.loc[:, "lower"]) / np.log(df.loc[:, "ppl"])
    df.loc[:, "score4"] = -np.log(df.loc[:, "window"])

    ## Deduplicate and select top-k.
    for column in [i for i in df.columns if i.startswith("score")]:
        ## First, we sort values.
        df = df.sort_values(by=column, ascending=False).reset_index(drop=True)

        ## BPE token-level similarity.
        top_k_token = []
        top_k_idx = []

        desc = f"🚗 Deduplicating (by={column})"
        with tqdm.tqdm(desc=desc, total=config.n_selected_samples) as pbar:
            for idx, row in df.iterrows():
                ## We only want top-k sentences.
                if len(top_k_token) >= config.n_selected_samples:
                    break

                ## Big O complexity: O(n(n-1)/2) where n is k.
                t = " ".join(
                    [
                        str(j)
                        for j in tok.encode(
                            row["text"], add_special_tokens=False
                        )
                    ]
                )
                if top_k_token == [] or all(
                    [
                        calculate_similarity(t.split(), token.split()) < 0.5
                        for token in top_k_token
                    ]
                ):
                    top_k_token.append(t)  ## save for comparison
                    top_k_idx.append(idx)  ## save for marking

                    ## Update probress bar.
                    pbar.update(1)

        ## Because there are many similar sentences,
        ## k-unique sentences may not be selected.
        df.loc[top_k_idx, f"{column}_top_k"] = np.arange(len(top_k_idx))

    ## Save when only local_rank is 0.
    ## Be careful not to dump both processes at the same time.
    if local_rank == 0:
        save_name = ".".join(
            [
                config.pretrained_model_name.replace("/", "_"),
                str(config.n_generated_samples),
                config.nowtime,
                "json",
            ]
        )
        save_results(df, save_name, config.assets)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
