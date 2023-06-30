import torch
import transformers
import deepspeed

import argparse
import logging
import tqdm

import numpy as np
import pandas as pd

from pathlib import Path
from typing import List

from src.score import GPTScorer
from src.utils import define_logger, print_config, remove_prefix_from_state_dict


LOGGER = logging.getLogger(__name__)


def define_argparser() -> argparse.Namespace:
    """Defines the arguments that will be used to execute the code.

    Returns:
        argparse.Namespace: A dictionary whose arguments can be called
    """
    p = argparse.ArgumentParser()

    ## Whether to load from a checkpoint.
    p.add_argument(
        "--load_from_checkpoint",
        action="store_true",
        help=" ".join(
            [
                "Whether to load from a checkpoint.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=" ".join(
            [
                "The root directory where the checkpoint is stored.",
                "Default=%(default)s",
            ]
        ),
    )

    ## Model and tokenizer.
    p.add_argument(
        "--pretrained_model_name",
        type=str,
        default="EleutherAI/gpt-neo-1.3B",
        help=" ".join(
            [
                "Name of the model you want to fine-tune.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--revision",
        type=str,
        default="main",
        help=" ".join(
            [
                "Revision of the pretrained model.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help=" ".join(
            [
                "The device on which the model will be loaded.",
                "Multi-gpu inference is not yet implemented.",
                "Default=%(default)s",
            ]
        ),
    )

    ## Generation.
    p.add_argument(
        "--n",
        type=int,
        default=10_000,
        help=" ".join(
            [
                "The number of texts you want to sample.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--k",
        type=int,
        default=100,
        help=" ".join(
            [
                "The number of texts you want to screen out of the sampled texts.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help=" ".join(
            [
                "Number of samples to process in one batch.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--do_sample",
        action="store_true",
        help=" ".join(
            [
                "Whether or not to use sampling;",
                "use greedy decoding otherwise.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--min_new_tokens",
        type=int,
        default=64,
        help=" ".join(
            [
                "The minimum numbers of tokens to generate.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help=" ".join(
            [
                "The maximum numbers of tokens to generate.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help=" ".join(
            [
                "If set to int > 0, all ngrams of that size",
                "can only occur once.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help=" ".join(
            [
                "If set to float < 1, only the smallest set of most probable",
                "tokens with probabilities that add up to top_p or higher are",
                "kept for generation.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=40,
        help=" ".join(
            [
                "The number of highest probability vocabulary tokens to keep",
                "for top-k-filtering.",
                "Default=%(default)s",
            ]
        ),
    )

    ## Debug.
    p.add_argument(
        "--detect_anomaly",
        action="store_true",
        help=" ".join(
            [
                "Enable anomaly detection for the autograd engine.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help=" ".join(
            [
                "Specifies the debugging mode.",
                "Default=%(default)s",
            ]
        ),
    )

    config = p.parse_args()
    return config


@torch.inference_mode()
def generate(config: argparse.Namespace, tok, model, prompt: str) -> np.ndarray:
    """One-step to generate text.

    Args:
        config (argparse.Namespace): A dictionary with configuration items
        tok (_type_): A tokenizer
        model (_type_): A model to generate samples
        prompt (str): Prompt that start a generated sequence

    Returns:
        np.ndarray: Generated samples in batches
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
        min_length=config.min_new_tokens + prompt_len,
        max_length=config.max_new_tokens + prompt_len,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
        top_p=config.top_p,
        top_k=config.top_k,
    )
    ## |gen_tokens| = (batch_size, length)

    ## Don't forget detaching from gpu into cpu.
    return gen_tokens.cpu().numpy()


def calculate_similarity(
    token1: List[int],
    token2: List[int],
    n_gram: int = 3,
) -> float:
    """Calculate trigram similarity: str1 (reference) vs str2 (hyphothesis).

    Args:
        token1 (List[int]): Reference sentence
        token2 (List[int]): Hypothesis sentence
        n_gram (int, optional): Arguments that determine overlap. Defaults to 3.

    Returns:
        float: n_gram similarity of two tokens
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


def save_results(config: argparse.Namespace, df: pd.DataFrame) -> None:
    """Save the extraction results to a CSV file.

    Args:
        config (argparse.Namespace): A dictionary with configuration items
        df (pd.DataFrame): A dataframe where the extraction results are stored
    """
    ## Save the total results.
    fname = (
        f"{Path(config.checkpoint_path).parent.name}.csv"
        if config.load_from_checkpoint
        else "naive.csv"
    )
    save_path = Path(config.checkpoint_path).parent.parent / Path(fname)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(save_path, encoding="utf-8", index=False, header=True)
    print(f"Results save to {save_path}")


def main(config: dict) -> None:
    ## Print arguments.
    print_config(config)

    ## Set logger.
    define_logger(config.debug)

    ## Force a build of cpu Adam in a Python shell.
    ## See: https://github.com/microsoft/DeepSpeed/issues/1846
    deepspeed.ops.op_builder.CPUAdamBuilder().load()

    ## See: https://sebastianraschka.com/blog/2023/llm-mixed-precision.html
    torch.set_float32_matmul_precision("medium")

    ## Auto-detect error.
    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    ## Load tokenizer and model.
    ## See: https://huggingface.co/EleutherAI/gpt-neo-2.7B
    tok = transformers.AutoTokenizer.from_pretrained(
        config.pretrained_model_name,
        revision=config.revision,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name,
        revision=config.revision,
        pad_token_id=tok.eos_token_id,
        torch_dtype="auto",  ## loaded as torch.float32 (not fp16)
        ## The argument 'low_cpu_mem_use=True'
        ## may cause RuntimeError: Tensors are not contiguous ...
        # low_cpu_mem_usage=True,
    )

    ## Load state_dict.
    if config.load_from_checkpoint:
        state_dict = torch.load(config.checkpoint_path)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            state_dict = remove_prefix_from_state_dict(state_dict)
            model.load_state_dict(state_dict)

    ## Don't forget turn-on evaluation mode.
    _ = model.eval()
    if config.device.startswith("cuda") and torch.cuda.is_available():
        model = model.to(device=config.device, non_blocking=True)

    ## ========== ========== ==========
    ## TEXT SAMPLING
    ## ========== ========== ==========
    tokens = np.zeros((0, config.max_new_tokens + 1), dtype=np.int32)
    with tqdm.tqdm(total=config.n, desc="Generating Texts") as pbar:
        while True:
            ## Generate sentences with one batch.
            prompt = tok.eos_token
            gen_tokens = generate(config, tok, model, prompt=prompt)

            ## Truncate if we create more than the desired numbers.
            if len(tokens) + len(gen_tokens) > config.n:
                gen_tokens = gen_tokens[: config.n - len(tokens)]

            ## Gather it.
            tokens = np.concatenate([tokens, gen_tokens], axis=0)

            ## Update progressbar.
            pbar.update(len(gen_tokens))

            if len(tokens) >= config.n:
                break

    ## ========== ========== ==========
    ## MEMBERSHIP INFERENCE
    ## ========== ========== ==========
    p = []  ## perplexy (PPL)
    z = []  ## zlib entropy
    # w = []  ## PPL of sliding windows

    scorer = GPTScorer(tok=tok)

    with tqdm.tqdm(total=len(tokens), desc="Inferring Membership") as pbar:
        num_batches = int(np.ceil(len(tokens) / config.batch_size))

        for i in range(num_batches):
            ## Get a mini-batch.
            ##  - |batch| = (batch_size, min_length=max_length)
            sp = i * config.batch_size
            ep = min((i + 1) * config.batch_size, len(tokens))

            with torch.inference_mode():
                ## Convert batch into logits and the labels.
                labels = torch.Tensor(tokens[sp:ep]).long().to(model.device)
                logits = model(input_ids=labels, return_dict=True).logits

                p_ = scorer.perplexity(logits=logits, labels=labels)
                z_ = scorer.zlib_entropy(labels=labels)
                # w_ = scorer.window_perplexity(
                #     batch=batch,
                #     window_size=config.window_size,
                #     stride=config.stride,
                # )

            ## Gather it.
            p = np.concatenate([p, p_], axis=0)
            z = np.concatenate([z, z_], axis=0)
            # w = np.concatenate([w, w_], axis=0)

            ## Update progress bar.
            pbar.update(len(labels))

    ## Save the scores.
    df = pd.DataFrame(
        {
            "text": tok.batch_decode(tokens, skip_special_tokens=True),
            "token": [" ".join([str(j) for j in i]) for i in tokens],
            "ppl": p,
            "zlib_entropy": z,
            # "ppl_window": w,
        }
    )

    ## Scoring.
    ##  - Score 1: only pseudo perplexity (lower best -> multiply -1 to higher best)
    ##  - Score 2: zlib entropy / pseudo perpledixy (higher best)
    df.loc[:, "score1"] = -np.log(df.loc[:, "ppl"])
    df.loc[:, "score2"] = df.loc[:, "zlib_entropy"] / np.log(df.loc[:, "ppl"])
    # df.loc[:, "score3"] = -np.log(df.loc[:, "ppl_window"])

    ## ========== ========== ==========
    ## VERIFICATION
    ## ========== ========== ==========

    ## De-duplicating.
    for column in [i for i in df.columns if i.startswith("score")]:
        ## First, we sort values.
        df = df.sort_values(by=column, ascending=False).reset_index(drop=True)

        ## BPE token-level similarity.
        top_k_token = []
        top_k_idx = []

        desc = f"Deduplicating (by={column})"
        with tqdm.tqdm(desc=desc, total=config.k) as pbar:
            for idx, row in df.iterrows():
                ## We only want top-k sentences.
                if len(top_k_token) >= config.k:
                    break

                ## Big O complexity: O(n(n-1)/2) where n is k.
                if top_k_token == [] or all(
                    [
                        calculate_similarity(
                            row["token"].split(), token.split()
                        )
                        < 0.5
                        for token in top_k_token
                    ]
                ):
                    top_k_token.append(row["token"])  ## save for comparison
                    top_k_idx.append(idx)  ## save for marking

                    ## Update probress bar.
                    pbar.update(1)

        ## Because there are many similar sentences,
        ## k-unique sentences may not be selected.
        df.loc[top_k_idx, f"{column}_top_k"] = np.arange(len(top_k_idx))

    ## Drop a column.
    df = df.drop(columns=["token"])

    ## Save.
    save_results(config, df)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
