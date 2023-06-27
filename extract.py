import torch
import transformers
import deepspeed

import argparse
import easydict
import logging
import pprint
import tqdm
import yaml

import numpy as np
import pandas as pd

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

from src.rl_lightning import MinimumRiskTrainingModule
from src.score import GPTScorer
from src.utils import define_logger, remove_prefix_from_state_dict


LOGGER = logging.getLogger(__name__)


def define_config(fname: str = "assets/extract_config.yaml") -> dict:
    ## Load yaml configuration file.
    with open(fname) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = easydict.EasyDict(config)
    return config


def get_tokenizer_and_model(config: argparse.Namespace) -> tuple:
    ## See: https://huggingface.co/kakaobrain/kogpt
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name,
        revision=config.revision,
        # bos_token="[BOS]",
        # eos_token="[EOS]",
        # unk_token="[UNK]",
        # pad_token="[PAD]",
        # mask_token="[MASK]",
    )
    LOGGER.debug(f"Tokenizer loaded: {config.pretrained_model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name,
        revision=config.revision,
        pad_token_id=tokenizer.eos_token_id,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    n_params = (
        sum([p.numel() for p in model.parameters()]) / 10**9
    )  ## billion
    LOGGER.debug(
        f"Weights loaded: {config.pretrained_model_name} (# params: {n_params:.2f}B)"
    )

    return tokenizer, model


@torch.inference_mode()
def generate(config: dict, tok, model, prompt: str) -> List[str]:
    ## Encode it.
    tokens = tok.encode(prompt, return_tensors="pt").repeat(
        config.batch_size, 1
    )
    tokens = tokens.to(device=config.device, non_blocking=True)

    prompt_len = tokens.size(1)
    assert prompt_len == 1

    ## Generate texts from tokens.
    gen_tokens = model.generate(
        tokens,
        do_sample=True,
        min_length=config.min_new_tokens + prompt_len,
        max_length=config.max_new_tokens + prompt_len,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
        top_p=config.top_p,
        top_k=config.top_k,
    )

    ## Don't forget detaching from gpu into cpu.
    ## |gen_tokens| = (batch_size, max_length=min_length)
    return gen_tokens.cpu().numpy()


def calculate_similarity(
    token1: List[int], token2: List[int], n_gram: int = 3
) -> bool:
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


def save_results(config: dict, df: pd.DataFrame) -> None:
    ## Save the total results.
    if config.load_from_checkpoint:
        save_path = Path(config.checkpoint_path).parent / "{}.csv".format(
            Path(config.checkpoint_path).name
        )
    else:
        save_path = Path(config.checkpoint_path).parent / Path("naive.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(save_path, encoding="utf-8", index=False, header=True)
    LOGGER.debug(f"Results save to {save_path}")


def main(config: dict) -> None:
    def print_config(config: dict) -> None:
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))

    print_config(config)

    ## Set logger.
    define_logger(config.debug)

    ## Force a build of cpu Adam in a Python shell.
    ## See: https://github.com/microsoft/DeepSpeed/issues/1846
    deepspeed.ops.op_builder.CPUAdamBuilder().load()

    ## See:
    ##  - https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    ##  - https://sebastianraschka.com/blog/2023/llm-mixed-precision.html
    torch.set_float32_matmul_precision("medium")

    ## Auto-detect error.
    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    ## Load tokenizer, model, and lightning module.
    tok = transformers.AutoTokenizer.from_pretrained(
        config.pretrained_model_name,
        revision=config.revision,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name,
        revision=config.revision,
        pad_token_id=tok.eos_token_id,
        torch_dtype=torch.float16,
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
    w = []  ## PPL of sliding windows

    scorer = GPTScorer(tok=tok)

    with tqdm.tqdm(total=len(tokens), desc="Inferring Membership") as pbar:
        num_batches = int(np.ceil(len(tokens) / config.batch_size))

        for i in range(num_batches):
            ## Get a mini-batch.
            ##  - |batch| = (batch_size, min_length=max_length)
            sp = i * config.batch_size
            ep = min((i + 1) * config.batch_size, len(tokens))

            labels = torch.LongTensor(tokens[sp:ep]).to(model.device)
            print(labels, labels.size())
            print(tok.batch_decode(labels))
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
    config = define_config()
    main(config)
