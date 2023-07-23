import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import zlib

from collections import OrderedDict
from typing import Dict, List

import numpy as np


class ScoreFunction(object):
    def __init__(
        self,
        tok: AutoTokenizer,
        model: AutoModelForCausalLM,
        device: int,
        mi_metrics: List[str],
    ):
        super(ScoreFunction, self).__init__()

        self.tok = tok
        self.model = model
        self.device = device
        self.mi_metrics = mi_metrics

        self.pad_token_id = tok.pad_token_id
        self.ignore_index = -100
        self.window_size = 50
        self.stride = 8

    def __call__(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Perform scoring for cross entropy loss by default and set MI metrics.

        Args:
            texts (List[str]): List of texts
                - |texts| = (batch_size,)

        Returns:
            Dict[str, np.ndarray]: Scores with MI metrics
        """
        ## Result.
        rslt = OrderedDict({})

        ## Tokenize.
        tokens = self.tok(texts, padding=True, return_tensors="pt").input_ids

        ## Calculate cross entropy loss.
        if not "ce_loss" in self.mi_metrics:
            raise AssertionError("Cross entropy loss is a default metric.")

        ce_loss = self.ce_loss_without_reduction(tokens)
        rslt["ce_loss"] = ce_loss

        ## Calculate perplexity.
        if "ppl" in self.mi_metrics:
            rslt["ppl"] = self.perplexity(ce_loss)

        ## Calculate zlib.
        if "zlib" in self.mi_metrics:
            rslt["zlib"] = self.zlib_entropy(texts)

        ## Calculate lowercase.
        if "lower" in self.mi_metrics:
            lower_texts = [t.lower() for t in texts]
            lower_tokens = self.tok(
                lower_texts,
                padding=True,
                return_tensors="pt",
            ).input_ids
            ## |lower_tokens| = (batch_size, unknown)

            lower_ce_loss = self.ce_loss_without_reduction(lower_tokens)
            rslt["lower"] = self.perplexity(lower_ce_loss)

        ## Calculate window.
        if "window" in self.mi_metrics:
            rslt["window"] = self.window_perplexity(tokens)

        return rslt

    @torch.inference_mode()
    def ce_loss_without_reduction(self, tokens: torch.Tensor) -> np.ndarray:
        """Forward calculates the loss without performing any reductions.

        Args:
            tokens (torch.Tensor): Encoded words
                - |tokens| = (batch_size, 1 + length)

        Returns:
            np.ndarray: Cross entropy losses without reduction
                - |loss| = (batch_size,)
        """
        ## Calculate labels and logits.
        labels = tokens.to(device=self.device)
        logits = self.model(input_ids=labels, return_dict=True).logits
        ## |labels| = (batch_size, 1 + length, vocab_size)
        ## |logits| = (batch_size, 1 + length, vocab_size)

        ## Set ignore index.
        labels[labels == self.pad_token_id] = self.ignore_index

        ## Move labels to correct device to enable model parallelism.
        labels = labels.to(device=self.device)

        ## Shift so that tokens < n predict n.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ## |shift_logits| = (batch_size, length, n_vocabs)
        ## |shift_labels| = (batch_size, length)

        batch_size = shift_logits.size(0)
        output_size = shift_logits.size(-1)

        ## Flatten the tokens without reduction.
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = (
            loss_fct(
                shift_logits.view(-1, output_size),
                shift_labels.view(-1),
            )
            .view(batch_size, -1)
            .mean(dim=-1)
        )
        loss = loss.detach().cpu().numpy()
        ## |loss| = (batch_size,)
        return loss

    @torch.inference_mode()
    def perplexity(self, ce_loss: np.ndarray) -> np.ndarray:
        """Computes the perplexity for each element in the batch.

        Args:
            ce_loss (np.ndarray): Cross entropy losses without reduction
                - |ce_loss| = (batch_size,)

        Returns:
            np.ndarray: Perplexities without reduction
                - |ppl| = (batch_size,)
        """
        ppl = np.exp(ce_loss)
        ## |ppl| = (batch_size,)
        return ppl

    def zlib_entropy(self, texts: List[str]) -> np.ndarray:
        """Compute the zlib entropy for each element in the label.

        Args:
            texts (List[str]): List of texts
                - |texts| = (batch_size,)

        Returns:
            np.ndarray: zlib entropy for each element
                - |z| = (batch_size,)
        """
        z = [len(zlib.compress(bytes(t, encoding="utf-8"))) for t in texts]
        z = np.array(z)
        ## |z| = (batch_size,)
        return z

    @torch.inference_mode()
    def window_perplexity(self, tokens: torch.Tensor) -> np.ndarray:
        """Calculate the slicing window perplexity of tokens.

        Args:
            tokens (torch.Tensor): Encoded words
                - |tokens| = (batch_size, 1 + length)

        Returns:
            np.ndarray: Non-reduced window perplexities
                - |ppl| = (batch_size,)
        """
        ## Results.
        ppls = []

        for sp in range(0, tokens.size(1) - self.window_size, self.stride):
            ## Slice.
            tokens_ = tokens[:, sp : sp + self.window_size]

            ## Calculate cross entropy loss.
            ce_loss = self.ce_loss_without_reduction(tokens_)
            ppl = self.perplexity(ce_loss)
            ## |ce_loss| = (batch_size,)
            ## |ppl| = (batch_size,)

            ## Gather it.
            ppls.append(ppl)

        ## Calculate the minimum ppl.
        ppl = np.min(np.stack(ppls), axis=0)
        ## |ppl| = (window_size, batch_size) -> (batch_size,)
        return ppl
