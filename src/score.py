import torch

import zlib

import numpy as np


class GPTScorer:
    def __init__(self, tok, model):
        super(GPTScorer, self).__init__()

        self.tok = tok
        self.model = model

        self.ignore_index = -100

    # @torch.inference_mode()
    def _forward_without_reduction(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        ## See forward function in GPT2LMHeadModel: http://bit.ly/3GDFDUq
        outputs = self.model.transformer(input_ids)
        hidden_states = outputs[0]

        lm_logits = self.model.lm_head(hidden_states)

        ## Shift so that tokens < n predict n.
        ##  - |shift_logits| = (batch_size, length - 1, n_vocabs)
        ##  - |shift_labels| = (batch_size, length - 1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        ## Flatten the tokens without reduction.
        ##  - |loss| = (batch_size, length - 1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(-1, shift_labels.size(-1))

        ## Return.
        return loss

    # @torch.inference_mode()
    def ce_loss(self, batch: torch.Tensor) -> torch.Tensor:
        ## Calcualte perplexity.
        device = batch.device
        input_ids = batch.clone().to(device=device, non_blocking=True)
        labels = torch.cat(
            [batch.clone()[:, 1:], batch.clone()[:, :1]], dim=1
        ).to(device=device, non_blocking=True)

        ## Forward and average it by token dimension.
        loss = self._forward_without_reduction(
            input_ids=input_ids, labels=labels
        )
        loss = loss.mean(dim=-1)
        ## ppl = np.exp(loss.detach().cpu().numpy())

        return loss

    def zlib_entropy(self, batch: np.ndarray) -> np.ndarray:
        ## Calculate zlib entropy.
        ##  - |entropy| = (batch_size,)
        entropy = [
            len(zlib.compress(bytes(s, encoding="utf-8")))
            for s in self.tok.batch_decode(batch, skip_special_tokens=True)
        ]
        entropy = torch.Tensor(entropy)  ## torch.float32

        return entropy

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
