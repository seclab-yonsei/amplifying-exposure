import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse
import logging
import os
import tqdm

import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Tuple

from src.score import ScoreFunction
from src.utils import (
    define_logger,
    load_results,
    print_config_rank_0,
    print_rank_0,
    save_results,
)

LOGGER = logging.getLogger(__name__)


def define_argparser() -> argparse.Namespace:
    """Defines the arguments that will be used to execute the code.

    Returns:
        argparse.Namespace: A dictionary whose arguments can be called
    """
    p = argparse.ArgumentParser()

    ## Continue to extract.
    p.add_argument(
        "--load_file",
        action="store_true",
        help="",
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

    ## Automated arguments.
    if config.assets in config.pretrained_model_name:
        model_name = "_".join(config.pretrained_model_name.split("/")[-2:])
    else:
        model_name = config.pretrained_model_name.replace("/", "_")

    config.save_name = "{}.{}.{}.json".format(
        model_name,
        config.n_generated_samples,
        config.nowtime,
    )
    config.save_path = Path(
        config.assets,
        model_name.replace("_actor_ema", ""),
        config.save_name,
    )

    return config


def get_tokenizer_and_model(
    pretrained_model_name: str,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Get a tokenizer and a model.

    Args:
        pretrained_model_name (str): A pretrained model name

    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM]: A tokenizer and a model
    """
    ## Load a tokenizer.
    tok = AutoTokenizer.from_pretrained(pretrained_model_name)
    tok.pad_token = tok.eos_token  ## Enable padding.
    tok.pad_token_id = tok.eos_token_id

    ## Load a model.
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name,
        pad_token_id=tok.eos_token_id,
    )

    return tok, model


def generate_texts(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: int,
    batch_size: int,
    do_sample: bool = True,
    min_new_tokens: int = 256,
    max_new_tokens: int = 256,
    no_repeat_ngram_size: int = 3,
    top_p: float = 0.95,
    top_k: int = 40,
    temperature: float = 1.0,
    n_generated_samples: int = 100_000,
    disable_tqdm: bool = False,
) -> pd.DataFrame:
    """Generate texts.

    Args:
        tok (AutoTokenizer): Tokenizer function
        model (AutoModelForCausalLM): Causal LM to generate text
        device (int): The device number on which the Model is loaded
        batch_size (int): Number of samples to process in one batch
        do_sample (bool, optional): Whether or not to use sampling; use greedy decoding otherwise. Defaults to True.
        min_new_tokens (int, optional): The minimum numbers of tokens to generate. Defaults to 256.
        max_new_tokens (int, optional): The maximum numbers of tokens to generate. Defaults to 256.
        no_repeat_ngram_size (int, optional): If set to int > 0, all ngrams of that size can only occur once. Defaults to 3.
        top_p (float, optional): Top-p sampling coefficient. Defaults to 0.95.
        top_k (int, optional): Top-k sampling coefficient. Defaults to 40.
        temperature (float, optional): The value used to modulate the next token probabilities. Defaults to 1.0.
        n_generated_samples (int, optional): The number of texts you want to sample. Defaults to 100_000.
        disable_tqdm (bool, optional): Whether to disable tqdm progress bar. Defaults to False.

    Returns:
        pd.DataFrame: Texts generated from the model
    """
    ## Outputs.
    out = []

    with tqdm.tqdm(
        total=n_generated_samples,
        desc="[+] Generating texts",
        disable=disable_tqdm,
    ) as pbar:
        ## Calcualate total iterations.
        n_batches = int(np.ceil(n_generated_samples / batch_size))
        for _ in range(n_batches):
            ## Generate sentences with one batch.
            prompts = torch.tensor(tok.eos_token_id, dtype=torch.int32)
            # prompts = tok.encode(
            #     "",
            #     return_tensors="pt",
            #     add_special_tokens=True,
            # )
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

            ## Truncate if we create more than the desired numbers.
            if len(out) + len(tokens) > n_generated_samples:
                tokens = tokens[: n_generated_samples - len(out)]

            ## Detokenize and calculate the number of tokens per sample.
            texts = tok.batch_decode(tokens, skip_special_tokens=True)
            n_tokens = (tokens != tok.pad_token_id).sum(dim=1)

            ## Gather it.
            out += [
                {"text": t, "n_tokens": int(n), "n_words": len(t.split(" "))}
                for t, n in zip(texts, n_tokens)
            ]

            ## Update progressbar.
            pbar.update(len(texts))

            if len(out) >= n_generated_samples:
                break

    out = pd.DataFrame(out)
    return out


def score_texts(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: int,
    mi_metrics: List[str],
    out: pd.DataFrame,
    batch_size: int,
    disable_tqdm: bool = False,
) -> pd.DataFrame:
    """Score texts.

    Args:
        tok (AutoTokenizer): Tokenizer function
        model (AutoModelForCausalLM): Causal LM to generate text
        device (int): The device number on which the Model is loaded
        mi_metrics (List[str]): Membership inference metrics we want to use
        out (pd.DataFrame): Texts generated from the model
        batch_size (int): Number of samples to process in one batch
        disable_tqdm (bool, optional): Whether to disable tqdm progress bar. Defaults to False.

    Returns:
        pd.DataFrame: Texts generated from the model and their scores
    """
    ## Add new columns.
    out.loc[:, mi_metrics] = np.nan

    ## Membership inference function.
    score_fn = ScoreFunction(
        tok=tok,
        model=model,
        device=device,
        mi_metrics=mi_metrics,
    )

    with tqdm.tqdm(
        total=len(out),
        desc="[+] Calculating {}".format(", ".join(mi_metrics)),
        disable=disable_tqdm,
    ) as pbar:
        ## Calcualate total iterations.
        n_batches = int(np.ceil(len(out) / batch_size))
        for i in range(n_batches):
            ## Get a mini-batch from start to end point.
            ## Note that df.loc[sp:ep] != df.loc[range(sp, ep)] :(
            sp = i * batch_size
            ep = min((i + 1) * batch_size, len(out))
            batch = out.loc[range(sp, ep), "text"].values.tolist()
            ## |batch|: List[str] = (batch_size,)

            ## Get membership inference metrics.
            scores = score_fn(batch)
            scores = pd.DataFrame(scores)

            ## Save the results.
            out.loc[range(sp, ep), scores.keys()] = scores.values

            ## Update progress bar.
            pbar.update(len(scores))

    ## Assert there are no nan values.
    assert out.isna().sum().sum() == 0
    return out


def deduplicate_texts(
    tok: AutoTokenizer,
    out: pd.DataFrame,
    n_selected_samples: int = 100,
    disable_tqdm: bool = False,
) -> pd.DataFrame:
    """Deduplicate texts and select top-k.

    Args:
        tok (AutoTokenizer): Tokenizer function
        out (pd.DataFrame): Texts generated from the model and their scores
        n_selected_samples (int): The number of texts you want to screen out of the sampled texts
        disable_tqdm (bool, optional): Whether to disable tqdm progress bar. Defaults to False.

    Returns:
        pd.DataFrame: Texts generated from the model, their scores, and top-k
    """
    ## Scoring.
    ##  - Score 1: only perplexity (lower best -> multiply -1 to higher best)
    ##  - Score 2: zlib entropy / perpledixy (higher best)
    ##  - Score 3: lowercase perplexity ratio (higher best)
    ##  - Score 4: window perpledixy (lower best -> multiply -1 to higher best)
    out.loc[:, "score1"] = -np.log(out.loc[:, "ppl"])
    out.loc[:, "score2"] = out.loc[:, "zlib"] / np.log(out.loc[:, "ppl"])
    out.loc[:, "score3"] = np.log(out.loc[:, "lower"]) / np.log(
        out.loc[:, "ppl"]
    )
    out.loc[:, "score4"] = -np.log(out.loc[:, "window"])

    ## Trigram similarity function.
    def _calculate_similarity(
        token1: List[int],
        token2: List[int],
        n_gram: int = 3,
    ) -> bool:
        """Compute the n-gram similarity between two tokenized text.

        Args:
            token1 (List[int]): First tokens
            token2 (List[int]): Second tokens
            n_gram (int, optional): Overlap coefficient. Defaults to 3.

        Returns:
            bool: N-gram similarity between token1 and token2
        """
        ## Calculate trigram similarity: str1 (reference) vs str2 (hyphothesis).
        ## It is same as "Is string 1 is similar with string 2?"

        ## Occasionally, if the Text consists of 1 or 2 words, the trigram multiset
        ## will result in an empty list, resulting in a divided by zero error.
        ## To prevent this, we guarantee that the trigram multiset has at least one element.

        ## We need to remove a EOS token.
        n_gram_set = lambda x: [
            " ".join(x[i : i + n_gram])
            for i in range(1, max(len(x) - n_gram, 1), 1)
        ]

        s1 = n_gram_set(token1)
        s2 = n_gram_set(token2)

        ## Return true if str1 is similar (or duplicated) to str2 else false.
        ## It is not recommended to mark two strings as similar, trivially.
        return len([i for i in s1 if i in s2]) / len(s1)

    ## Deduplicate and select top-k.
    THRESHOLD = 0.5
    for column in [i for i in out.columns if i.startswith("score")]:
        ## First, we sort values.
        out = out.sort_values(by=column, ascending=False).reset_index(drop=True)

        ## BPE token-level similarity.
        top_k_token = []
        top_k_idx = []

        with tqdm.tqdm(
            total=n_selected_samples,
            desc=f"[+] Deduplicating (by={column})",
            disable=disable_tqdm,
        ) as pbar:
            for idx, row in out.iterrows():
                ## We only want top-k sentences.
                if len(top_k_token) >= n_selected_samples:
                    break

                ## Big O complexity: O(n(n-1)/2) where n is k.
                t = " ".join(
                    [
                        str(j)
                        for j in tok.encode(
                            row["text"],
                            add_special_tokens=False,
                        )
                    ]
                )
                if top_k_token == [] or all(
                    [
                        _calculate_similarity(t.split(), token.split())
                        < THRESHOLD
                        for token in top_k_token
                    ]
                ):
                    top_k_token.append(t)  ## save for comparison
                    top_k_idx.append(idx)  ## save for marking

                    ## Update probress bar.
                    pbar.update(1)

        ## Because there are many similar sentences,
        ## k-unique sentences may not be selected.
        out.loc[top_k_idx, f"{column}_top_k"] = np.arange(len(top_k_idx))

    return out


def main(config: argparse.Namespace) -> None:
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

    ## Print arguments.
    print_config_rank_0(config, LOCAL_RANK)

    ## Load tokenizer and model.
    tok, model = get_tokenizer_and_model(config.pretrained_model_name)
    ms = f"[+] Tokenizer and model are loaded: {config.pretrained_model_name}"
    print_rank_0(ms, LOCAL_RANK)
    print_rank_0(f"[+] tok: {tok}", LOCAL_RANK)
    print_rank_0(f"[+] model: {model}", LOCAL_RANK)

    ## Initialize deepspeed inference mode.
    ds_engine = deepspeed.init_inference(
        model=model,
        dtype=torch.half,
        tensor_parallel={"tp_size": WORLD_SIZE},
        ## It may cause error in OPT :(
        ## See: https://sooftware.io/neox_injection/
        # replace_with_kernel_inject=True,
    )
    ## Don't forget turn-on evaluation mode.
    ds_engine.module.eval()

    ## ========== TEXT GENERATION ==========
    if config.load_file:
        out = load_results(config.save_path)
        print_rank_0(f"[+] Results load from {config.save_path}", LOCAL_RANK)
    else:
        out = generate_texts(
            tok=tok,
            model=ds_engine.module,
            device=LOCAL_RANK,
            batch_size=config.batch_size,
            do_sample=config.do_sample,
            min_new_tokens=config.min_new_tokens,
            max_new_tokens=config.max_new_tokens,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            top_p=config.top_p,
            top_k=config.top_k,
            temperature=config.temperature,
            n_generated_samples=config.n_generated_samples,
            disable_tqdm=False if LOCAL_RANK <= 0 else True,
        )

    ## ========== TEXT RANKING ==========
    out = score_texts(
        tok=tok,
        model=ds_engine.module,
        device=LOCAL_RANK,
        mi_metrics=config.mi_metrics,
        out=out,
        batch_size=config.batch_size,
        disable_tqdm=False if LOCAL_RANK <= 0 else True,
    )

    ## ========== SCORING, DEDUPLICATING, SELECT TOP-K ==========
    if config.do_scoring:
        out = deduplicate_texts(
            tok=tok,
            out=out,
            n_selected_samples=config.n_selected_samples,
            disable_tqdm=False if LOCAL_RANK <= 0 else True,
        )
        config.save_path = Path(config.save_path).with_suffix(".extract.json")

    ## ========== SAVE TO DATAFRAME ==========
    if LOCAL_RANK <= 0:
        ## Save only one time in main process.
        save_results(out, config.save_path)
        print_rank_0(f"[+] Results save to {config.save_path}", LOCAL_RANK)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
