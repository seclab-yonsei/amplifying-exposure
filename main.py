import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

import argparse
import datetime
import easydict
import json
import logging
import re
import tqdm

import numpy as np

from pathlib import Path
from pytz import timezone
from sklearn.model_selection import train_test_split
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
        "--mask_filling_model_name",
        type=str,
        default="t5-large",  ## 770M
        help="Name of the model you want to fill mask.",
    )
    p.add_argument(
        "--device",
        type=str,
        nargs=2,
        default=["cuda:0", "cuda:1"],
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

    ## DetectGPT.
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

    ## Train & test split.
    p.add_argument(
        "--test_size",
        type=float,
        default=0.2,
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
def generate(config, tok, model, prompt: str = "") -> torch.Tensor:
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
    tokens = tok.encode(prompt, return_tensors="pt", add_special_tokens=True)
    tokens = tokens.repeat(config.batch_size, 1)
    tokens = tokens.to(device=model.device, non_blocking=True)
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


def tokenize_and_mask(config, gen_texts: List[dict]):
    ## Configurations.
    mask_string = "<<<mask>>>"
    threshold = 20

    ## Lambda.
    def _calculate_n_spans(tokens: List[str]):
        n_spans = (
            config.pct_words_masked
            * len(tokens)
            / (config.span_length + config.buffer_size * 2)
        )
        n_spans = int(np.ceil(n_spans))
        return n_spans

    ## Results.
    out = []

    ## Split and mask per text.
    desc = f"🚀 Split and Masking"
    for item in tqdm.tqdm(gen_texts, desc=desc):
        text = item.text
        perturb_texts = []

        ## Validation check: num words <= 20? continue;.
        if len(text.split(" ")) <= threshold:
            continue

        ## Generate n perturb samples.
        for i in range(config.n_perturb_samples):
            tokens = text.split(" ")

            n_spans = _calculate_n_spans(tokens)
            n_masks = 0

            while n_masks < n_spans:
                sp = np.random.randint(0, len(tokens) - config.span_length)
                ep = sp + config.span_length

                search_sp = max(0, sp - config.buffer_size)
                search_ep = min(len(tokens), ep + config.buffer_size)

                if mask_string not in tokens[search_sp:search_ep]:
                    tokens[sp:ep] = [mask_string]
                    n_masks += 1

            ## Replace each occurrence of mask_string with <extra_id_NUM>,
            ## where NUM increments
            num_filled = 0
            for idx, token in enumerate(tokens):
                if token == mask_string:
                    tokens[idx] = f"<extra_id_{num_filled}>"
                    num_filled += 1

            msg = f"num_filled {num_filled} != n_masks {n_masks}"
            assert num_filled == n_masks, msg

            perturb_text = " ".join(tokens)
            perturb_texts.append({"id": i, "masked_text": perturb_text})

        item.perturb_texts = perturb_texts
        out.append(item)

    ## Check.
    diff = len(gen_texts) - len(out)
    msg = f"🙄 {diff} sample(s) that were not properly sampled were dropped."
    if diff > 0:
        print(msg)

    return out


def count_masks(texts: List[str]):
    return [
        len([x for x in text.split() if x.startswith("<extra_id_")])
        for text in texts
    ]


# replace each masked span with a sample from T5 mask_model
def replace_masks(config, texts, mask_tok, mask_model):
    n_expected = count_masks(texts)
    stop_id = mask_tok.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tok(texts, return_tensors="pt", padding=True).to(
        mask_model.device
    )
    outputs = mask_model.generate(
        **tokens,
        do_sample=config.do_sample,
        min_new_tokens=config.min_new_tokens,
        max_new_tokens=config.max_new_tokens,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
        top_p=config.top_p,
        top_k=config.top_k,
        temperature=config.temperature,
        num_return_sequences=1,
        eos_token_id=stop_id,
    )
    return mask_tok.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts, mask_tok):
    pattern = re.compile(r"<extra_id_\d+>")

    # remove <pad> from beginning of each text
    texts = [
        x.replace(mask_tok.pad_token, "")
        .replace(mask_tok.eos_token, "")
        .strip()
        for x in texts
    ]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(" ") for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(
        zip(tokens, extracted_fills, n_expected)
    ):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def save_results(
    config,
    rslt: List[dict],
    train_pairs: List[dict],
    eval_pairs: List[dict],
) -> None:
    """Save the extraction results to a CSV file.

    Args:
        config (_type_): A dictionary with configuration items
        rslt (list): A list where the extraction results are stored as dict
    """
    ## Save the total results.
    kst = timezone("Asia/Seoul")
    nowtime = datetime.datetime.now(kst).strftime("%Y%m%d-%H%M%S")
    model_name = config.pretrained_model_name.replace("/", "_")
    n_samples = int(config.n // 1e3)

    fname = f"{model_name}.{n_samples:03d}k.{nowtime}.jsonl"  ## not json
    save_path = Path(config.assets, fname)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(rslt, f, indent=4)
        f.write("\n")

    print(f"Results save to {save_path}")

    train_fname = f"{model_name}.{n_samples:03d}k.{nowtime}.train-pairs.json"
    train_save_path = Path(config.assets, train_fname)
    eval_fname = f"{model_name}.{n_samples:03d}k.{nowtime}.eval-paris.json"
    eval_save_path = Path(config.assets, eval_fname)

    with open(train_save_path, "w", encoding="utf-8") as f:
        json.dump(train_pairs, f, indent=4)

    with open(eval_save_path, "w", encoding="utf-8") as f:
        json.dump(eval_pairs, f, indent=4)

    print(f"Results save to {train_save_path}")
    print(f"Results save to {eval_save_path}")


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

    ## Devices.
    config.main_device = config.device[0]
    config.aux_device = config.device[1]

    ## Load tokenizer and model.
    ## See: https://huggingface.co/EleutherAI/gpt-neo-2.7B
    tok = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name,
        pad_token_id=tok.eos_token_id,
    )
    ## Enable padding.
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

    ## Don't forget turn-on evaluation mode.
    _ = model.eval()
    if config.main_device.startswith("cuda") and torch.cuda.is_available():
        model = model.to(device=config.main_device, non_blocking=True)

    ## Text sampling.
    rslt = []
    with tqdm.tqdm(total=config.n, desc="🛴 Generating Texts") as pbar:
        while True:
            ## Generate sentences with one batch.
            prompt = ""
            gen_tokens = generate(config, tok, model, prompt=prompt)

            ## Truncate if we create more than the desired numbers.
            if len(rslt) + len(gen_tokens) > config.n:
                gen_tokens = gen_tokens[: config.n - len(rslt)]

            ## Detokenize and calculate the number of tokens per sample.
            gen_texts = tok.batch_decode(gen_tokens, skip_special_tokens=True)
            n_tokens = (gen_tokens != tok.pad_token_id).sum(dim=1)

            ## Gather it.
            rslt += [
                easydict.EasyDict(
                    {
                        "text": g,
                        "n_tokens": int(n),
                        "n_words": len(g.split(" ")),
                    }
                )
                for g, n in zip(gen_texts, n_tokens)
            ]

            ## Update progressbar.
            pbar.update(len(gen_texts))

            if len(rslt) >= config.n:
                break

    ## Membership inference.
    with tqdm.tqdm(total=len(rslt), desc="🚂 Calculating Loss") as pbar:
        n_batches = int(np.ceil(len(rslt) / config.batch_size))

        for i in range(n_batches):
            ## Get a mini-batch from start to end point.
            sp = i * config.batch_size
            ep = min((i + 1) * config.batch_size, len(rslt))

            batch = [rslt[j]["text"] for j in range(sp, ep, 1)]
            batch = tok(batch, padding=True, return_tensors="pt").input_ids

            ## Calculate negative log probabilies.
            ce_loss = inference(tok, model, batch)

            ## Save it.
            for j in range(sp, ep, 1):
                rslt[j]["loss"] = float(ce_loss[j - sp])

            ## Update progress bar.
            pbar.update(len(ce_loss))

    ## Calculate machine-generated probabilities.
    rslt = tokenize_and_mask(config, rslt)

    ## Calculate t5 probability.

    ## Load tokenizer and model.
    ## See: https://github.com/huggingface/transformers/issues/3985
    mask_model = AutoModelForSeq2SeqLM.from_pretrained(
        config.mask_filling_model_name,
    )
    mask_tok = AutoTokenizer.from_pretrained(
        config.mask_filling_model_name,
        model_max_length=512,
    )

    ## Don't forget turn-on evaluation mode.
    _ = mask_model.eval()
    # if config.aux_device.startswith("cuda") and torch.cuda.is_available():
    #     mask_model = mask_model.to(device=config.aux_device, non_blocking=True)

    desc = "🐢 Calculating Log Ratio of Perturbed Texts"
    with tqdm.tqdm(total=len(rslt), desc=desc) as pbar:
        n_batches = int(np.ceil(len(rslt) / config.batch_size))

        for i in range(n_batches):
            ## Get a mini-batch from start to end point.
            sp = i * config.batch_size
            ep = min((i + 1) * config.batch_size, len(rslt))

            for j in range(config.n_perturb_samples):
                batch = [
                    rslt[k]["perturb_texts"][j].masked_text
                    for k in range(sp, ep, 1)
                ]
                raw_fills = replace_masks(config, batch, mask_tok, mask_model)
                extracted_fills = extract_fills(raw_fills, mask_tok)
                perturbed_texts = apply_extracted_fills(batch, extracted_fills)

                ## Calculate loss.
                batch = tok(
                    perturbed_texts,
                    padding=True,
                    return_tensors="pt",
                ).input_ids
                ce_loss = inference(tok, model, batch)  ## not mask_model

                for k in range(sp, ep, 1):
                    rslt[k]["perturb_texts"][j].perturb_text = perturbed_texts[
                        k - sp
                    ]
                    rslt[k]["perturb_texts"][j].loss = float(ce_loss[k - sp])

            for j in range(sp, ep, 1):
                l = [
                    rslt[j]["perturb_texts"][k]["loss"]
                    for k in range(config.n_perturb_samples)
                ]
                ## Calculate perturbation discrepancy.
                d = rslt[j]["loss"] - np.mean(l)
                score = d / np.sqrt(np.std(l))

                ## Save.
                rslt[j]["score"] = score

            ## Update progress bar.
            pbar.update(len(batch))

    ## Calculate pairs.
    scores = np.array([rslt[i]["score"] for i in range(len(rslt))])
    indexs = np.arange(len(rslt))

    np.random.shuffle(indexs)
    if len(indexs) % 2:  ## make length to even
        indexs = indexs[:-1]

    scores = scores[indexs]
    scores = scores.reshape(-1, 2)
    indexs = indexs.reshape(-1, 2)

    ## Make pairs.
    pairs = []
    for (s1, s2), (i1, i2) in zip(scores, indexs):
        ## A larger log probability is fake text
        ##  == Smaller negative log probability is fake text
        chosen_idx, rejected_idx = (i1, i2) if s1 < s2 else (i2, i1)

        chosen = rslt[chosen_idx]["text"]
        rejected = rslt[rejected_idx]["text"]

        items = {
            "prompt": "Human: " + "" + " Assistant:",  ## empty prompt
            "chosen": chosen,
            "rejected": rejected,
        }
        pairs.append(items)

    ## Train & test split.
    train_pairs, eval_pairs = train_test_split(
        pairs,
        test_size=config.test_size,
        shuffle=True,
    )

    ## Save.
    save_results(config, rslt, train_pairs, eval_pairs)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
