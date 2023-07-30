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
from sklearn.model_selection import train_test_split
from typing import Tuple

from src.score import ScoreFunction
from src.utils import (
    define_logger,
    print_config_rank_0,
    print_rank_0,
    load_results,
    save_results,
    save_pairs,
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

    ## Inference.
    p.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Number of samples to process in one batch.",
    )

    ## DetectGPT.
    p.add_argument(
        "--n_perturbed_samples",
        type=int,
        default=3,
        help="Number of samples to perturb in one sample.",
    )

    ## Train & test split.
    p.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="",
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
    model_name = config.pretrained_model_name.replace("/", "_")
    config.save_name = "{}.{}.{}.{}.csv".format(
        model_name,
        config.n_generated_samples,
        config.nowtime,
        "perturb",
    )
    config.save_path = Path(config.assets, model_name, config.save_name)

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


def score_perturbed_texts(
    tok: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: int,
    out: pd.DataFrame,
    batch_size: int,
    disable_tqdm: bool = False,
) -> pd.DataFrame:
    """Compute the loss of the target LM for perturbed texts.

    Args:
        tok (AutoTokenizer): Tokenizer function
        model (AutoModelForCausalLM): Causal LM to generate text
        device (int): The device number on which the Model is loaded
        out (pd.DataFrame): A dataframe with generated texts and perturbated texts
        batch_size (int): Number of samples to process in one batch
        disable_tqdm (bool, optional): Whether to disable tqdm progress bar. Defaults to False.

    Returns:
        pd.DataFrame: A dataframe with loss of perturbated text
    """
    ## Membership inference function.
    score_fn = ScoreFunction(
        tok=tok,
        model=model,
        device=device,
        mi_metrics=["ce_loss"],
    )

    ## Get perturbed text columns using regex.
    p_cols = out.filter(like="perturbed_text_").columns

    ## Add new columns.
    p_loss_cols = p_cols + "_ce_loss"
    out.loc[:, p_loss_cols] = np.nan

    with tqdm.tqdm(
        total=len(out),
        desc="[+] Calculating loss of perturbed texts",
        disable=disable_tqdm,
    ) as pbar:
        ## Calculate total iterations.
        n_batches = int(np.ceil(len(out) / batch_size))
        for i in range(n_batches):
            ## Get a mini-batch from start to end point.
            sp = i * batch_size
            ep = min((i + 1) * batch_size, len(out))

            for p_col, p_loss_col in zip(p_cols, p_loss_cols):
                ## Set a mini-batch with perturbed texts.
                batch = out.loc[range(sp, ep), p_col].values.tolist()
                ## |batch|: List[str] = (batch_size,)

                ## Get cross entropy loss.
                scores = score_fn(batch)
                scores = pd.DataFrame(scores)

                ## Save the results.
                out.loc[range(sp, ep), p_loss_col] = scores.values

            ## Update progress bar.
            pbar.update(ep - sp)

    ## Assert there are no nan values.
    assert out.isna().sum().sum() == 0
    return out


def calculate_perturbation_discrepancy_score(out: pd.DataFrame) -> pd.DataFrame:
    """Calculate the perturbation discrepancy score for perturbed texts.

    Args:
        out (pd.DataFrame): A dataframe with loss of perturbated text

    Returns:
        pd.DataFrame: A dataframe with perturbation discrepancy score of each samples
    """
    ## Filter dataframes contains with cross entropy loss.
    ce_loss = out.filter(items=["ce_loss"])
    p_ce_loss = out.filter(like="perturbed_text_").filter(like="_ce_loss")

    ## Convert negative log likelihoods to log likelihoods as in paper.
    ce_loss = -ce_loss
    p_ce_loss = -p_ce_loss

    ## Calculate perturbation discrepancy score by z-score standarization.
    ## See: https://github.com/eric-mitchell/detect-gpt/issues/11
    d = pd.DataFrame({"score": ce_loss.loc[:, "ce_loss"]})
    d.loc[:, "score"] -= p_ce_loss.apply(np.mean, axis=1)
    d.loc[:, "score"] /= p_ce_loss.apply(np.std, axis=1)

    ## Append it.
    out.loc[:, "score"] = d.loc[:, "score"].values
    assert out.isna().sum().sum() == 0
    return out


def make_pairs(out: pd.DataFrame) -> pd.DataFrame:
    """Pair text for RLHF training.

    Args:
        out (pd.DataFrame): A dataframe with perturbation discrepancy score of each samples

    Returns:
        pd.DataFrame: A dataframe of dict containing prompt, chosen, and rejected
    """
    ## Sort by ascending.
    out = out.sort_values(by="score").reset_index(drop=True)

    ## Only left "score" and "text", and make it even.
    out = out.loc[range(int(len(out) // 2)), ["score", "text"]]

    ## Pair locally optimal so that the score difference is maximized.
    low = out.loc[range(int(len(out)) // 2)].reset_index(drop=True)
    high = out.loc[range(int(len(out)) // 2, len(out))].reset_index(drop=True)
    assert len(low) == len(high)

    ## Make it pairs.
    ## The prompt should be in the format of:
    ## >>> " Human: " + actual_prompt_sentence + " Assistant:"
    ## The chosen and rejected response should be in the format of:
    ## >>> " " + actual_response_sentence
    ## See: https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/utils/data/raw_datasets.py
    pairs = pd.DataFrame(
        {
            "prompt": [""] * len(low),
            "chosen": low.loc[:, "text"],
            "chosen_score": low.loc[:, "score"],
            "rejected": high.loc[:, "text"],
            "rejected_score": high.loc[:, "score"],
            "score_diff": high.loc[:, "score"] - low.loc[:, "score"],
        }
    )

    return pairs


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
    print_rank_0(
        f"[+] Tokenizer and model are loaded: {config.pretrained_model_name}",
        LOCAL_RANK,
    )
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

    ## ========== LOAD GENERATED TEXTS ==========
    out = load_results(config.save_path)
    print_rank_0(f"[+] Results load from {config.save_path}", LOCAL_RANK)

    ## Drop nan index.
    nan_idx = out.loc[out.isna().sum(axis=1) > 0].index
    out = out.drop(nan_idx).reset_index(drop=True)
    msg = f"[!] {len(nan_idx)} samples that have nan texts are dropped."
    print_rank_0(msg, LOCAL_RANK)

    ## ========== SCORE PERTURBED TEXTS ==========
    out = score_perturbed_texts(
        tok=tok,
        model=model,
        device=LOCAL_RANK,
        out=out,
        batch_size=config.batch_size,
        disable_tqdm=False if LOCAL_RANK <= 0 else True,
    )

    ## ========== CALCULATE PERTURBATION DISCREPANCY SCORES ==========
    out = calculate_perturbation_discrepancy_score(out=out)
    msg = f"[+] Perturbation discrepancy scores are calculated"
    print_rank_0(msg, LOCAL_RANK)

    ## ========== MAKE PAIRS AND SPLIT ==========
    pairs = make_pairs(out)
    print_rank_0(f"[+] Perturbation discrepancy scores are paired", LOCAL_RANK)

    tr_pairs, ev_pairs = train_test_split(
        pairs,
        test_size=config.test_size,
        shuffle=True,
    )
    print_rank_0(f"[+] Separated into train and eval Pairs", LOCAL_RANK)
    msg = f"[+]  - train.shape: {tr_pairs.shape}, eval.shape: {ev_pairs.shape}"
    print_rank_0(msg, LOCAL_RANK)

    ## ========== SAVE PAIRS  ==========
    if LOCAL_RANK <= 0:
        ## Save train and eval pairs to json only one time in main process..
        tr_pairs_path = Path(config.save_path).with_suffix(".pairs.train.json")
        ev_pairs_path = Path(config.save_path).with_suffix(".pairs.eval.json")
        save_pairs(tr_pairs, tr_pairs_path)
        save_pairs(ev_pairs, ev_pairs_path)
        print_rank_0(f"[+] Results save to {tr_pairs_path}", LOCAL_RANK)
        print_rank_0(f"[+] Results save to {ev_pairs_path}", LOCAL_RANK)

        ## Save total pairs with perturbation discrepancy scores to csv.
        config.save_path = Path(config.save_path).with_suffix(".detectgpt.csv")
        save_results(out, config.save_path)
        print_rank_0(f"[+] Results save to {config.save_path}", LOCAL_RANK)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
