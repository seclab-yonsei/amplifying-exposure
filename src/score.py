import torch

import zlib

import numpy as np


class GPTScorer:
    def __init__(self, tok):
        super(GPTScorer, self).__init__()
        self.tok = tok

    def ce_loss_without_reduction(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Forward calculates the loss without performing any reductions.

        Args:
            logits (torch.Tensor): A logit vector for each vocabulary
                - |logits| = (batch_size, length, vocab_size)
            labels (torch.Tensor): An answer vector
                - |labels| = (batch_size, length)

        Returns:
            torch.Tensor: Loss value per element without reduction applied
                - |loss| = (batch_size,)
        """
        ## Move labels to correct device to enable model parallelism.
        labels = labels.to(logits.device)

        ## Shift so that tokens < n predict n.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ## |shift_logits| = (batch_size, length-1, n_vocabs)
        ## |shift_labels| = (batch_size, length-1)

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
        ## |loss| = (batch_size,)
        return loss

    @torch.inference_mode()
    def perplexity(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> np.ndarray:
        """Computes the perplexity for each element in the batch.

        Args:
            logits (torch.Tensor): A logit vector for each vocabulary
                - |logits| = (batch_size, length, vocab_size)
            labels (torch.Tensor): An answer vector
                - |labels| = (batch_size, length)

        Returns:
            np.ndarray: Perplexity per element without reduction applied
                - |ppl| = (batch_size,)
        """
        ## Forward and average it by token dimension.
        loss = self.ce_loss_without_reduction(logits=logits, labels=labels)
        ## |loss| = (batch_size,)

        ppl = np.exp(loss.detach().cpu().numpy())
        ## |ppl| = (batch_size,)
        return ppl

    @torch.inference_mode()
    def zlib_entropy(self, labels: torch.Tensor) -> torch.Tensor:
        """Compute the zlib entropy for each element in the label.

        Args:
            labels (torch.Tensor): An answer vector
                - |labels| = (batch_size, length)

        Returns:
            torch.Tensor: zlib entropy for each element
                - |z| = (batch_size,)
        """
        z = torch.FloatTensor(
            [
                len(zlib.compress(bytes(sent, encoding="utf-8")))
                for sent in self.tok.batch_decode(
                    labels, skip_special_tokens=True
                )
            ]
        )
        ## |z| = (batch_size,)
        return z

    """
    @torch.inference_mode()
    def zlib_entropy_ratio(self, gen_tokens: torch.Tensor) -> torch.Tensor:
        ##  - |gen_tokens| = (batch_size, length)
        ##  - |zlib_entropy| = (batch_size,)
        b_sents = [
            bytes(sent, encoding="utf-8")
            for sent in self.tok.batch_decode(
                gen_tokens, skip_special_tokens=True
            )
        ]
        zlib_entropy_ratio = [
            len(zlib.compress(sent)) / len(sent) for sent in b_sents
        ]

        ## Dtype: long -> float
        zlib_entropy_ratio = torch.FloatTensor(zlib_entropy_ratio)

        return zlib_entropy_ratio
    """

    """
    def window_perplexity(
        self,
        batch: np.ndarray,
        window_size: int = 50,
        stride: int = 8,
    ) -> np.ndarray:
        ppls = []
        for sp in range(0, batch.shape[1] - window_size, stride):
            ppl = self.perplexity(batch[:, sp : sp + window_size])
            ppls.append(ppl)

        ## Calculate the minimum ppl.
        ##  - |ppl| = (batch_size,)
        ppl = np.min(np.stack(ppls), axis=0)

        return ppl
    """
