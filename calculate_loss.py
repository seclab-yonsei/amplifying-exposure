import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import argparse
import json
import logging
import tqdm

import numpy as np

from operator import itemgetter
from pathlib import Path
from pytz import timezone
from typing import List

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
        "--save_file",
        type=str,
        required=True,
        help="File name that you want to calculate loss of each samples.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Number of samples to process in one batch.",
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


def load_file(config) -> List[dict]:
    fpath = Path(config.assets, config.save_file)

    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()


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
    fname = Path(config.save_file).with_suffix(".jsonl")  ## txt to csv
    save_path = Path(config.assets, fname)
    save_path.mkdir(parents=True, exist_ok=True)

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

    ## Load texts.
    rslt = load_file(config)

    ## Membership inference.
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
