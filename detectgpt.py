import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse
import logging
import os
import tqdm

import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List

from src.dataset import make_pairs
from src.score import ScoreFunction
from src.utils import define_logger, print_config, load_results, save_results

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
    return config


def detectgpt_score(loss: float, perturbed_loss: List[float]) -> float:
    ## Calculate perturbation discrepancy.
    d = loss - np.mean(perturbed_loss)
    ## |d| = (1,)

    ## Normalize.
    score = d / np.sqrt(np.std(perturbed_loss))
    ## |score| = (1,)

    return score


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

    ## Load results.
    rslt, save_path = load_results(
        nowtime=config.nowtime,
        assets=config.assets,
        suffix="perturb",
    )

    ## Membership inference.
    score_fn = ScoreFunction(
        tok,
        ds_engine.module,
        device=local_rank,
        mi_metrics=["ce_loss"],
    )

    ## Calculate cross entropy ratio.
    desc = "🐢 Calculating Log Ratio of Perturbed Texts"
    with tqdm.tqdm(total=len(rslt), desc=desc) as pbar:
        n_batches = int(np.ceil(len(rslt) / config.batch_size))

        for i in range(n_batches):
            ## Get a mini-batch from start to end point.
            sp = i * config.batch_size
            ep = min((i + 1) * config.batch_size, len(rslt))

            ## Calculate perturbed texts' losses.
            for j in range(config.n_perturbed_samples):
                ## Gather perturbed texts.
                perturbed_texts = [
                    rslt[k]["perturbed_texts"][j]["perturbed_text"]
                    for k in range(sp, ep, 1)
                ]

                ## Get cross entropy loss.
                rslt_ = score_fn(perturbed_texts)
                ce_loss = rslt_["ce_loss"]

                ## Keep perturbed text's loss.
                for idx, k in enumerate(range(sp, ep, 1)):
                    rslt[k]["perturbed_texts"][j]["ce_loss"] = float(
                        ce_loss[idx]
                    )

            ## Calculate DetectGPT's scores.
            for j in range(sp, ep, 1):
                ## Gather all perturbed losses.
                perturbed_loss = [
                    rslt[j]["perturbed_texts"][k]["ce_loss"]
                    for k in range(config.n_perturbed_samples)
                ]
                score = detectgpt_score(rslt[j]["ce_loss"], perturbed_loss)

                ## Keep it.
                rslt[j]["score"] = score

            ## Update progress bar.
            pbar.update(ep - sp)

    ## Make pairs.
    scores = np.array([rslt[i]["score"] for i in range(len(rslt))])
    texts = [rslt[i]["text"] for i in range(len(rslt))]

    pairs = make_pairs(scores=scores, texts=texts)

    ## Train & test split.
    train_pairs, eval_pairs = train_test_split(
        pairs,
        test_size=config.test_size,
        shuffle=True,
    )

    ## Save results.
    train_pairs_path = Path(save_path).with_suffix(".pairs.train.json")
    eval_pairs_path = Path(save_path).with_suffix(".pairs.eval.json")

    if local_rank == 0:
        save_results(rslt, Path(save_path).name, config.assets)
        save_results(train_pairs, Path(train_pairs_path).name, config.assets)
        save_results(eval_pairs, Path(eval_pairs_path).name, config.assets)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
