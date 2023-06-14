import torch

import zlib

import numpy as np


class GPTScorer:
    def __init__(self, tok, model):
        super(GPTScorer, self).__init__()

        self.tok = tok
        self.model = model

        self.ignore_index = -100

    def _forward_without_reduction(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        ##  - |gen_tokens| = (batch_size, length)

        ## See forward function in GPT2LMHeadModel:
        ##  - https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L1055C9-L1126
        transformer_outputs = self.model.transformer(input_ids)
        hidden_states = transformer_outputs[0]

        ## Set device for model parallelism.
        if self.model.model_parallel:
            torch.cuda.set_device(self.model.transformer.first_device)
            hidden_states = hidden_states.to(self.model.lm_head.weight.device)

        lm_logits = self.model.lm_head(hidden_states)

        ## Move labels to correct device to enable model parallelism.
        ##  - |labels| = (batch_size, length)
        labels = labels.to(lm_logits.device)

        ## Shift so that tokens < n predict n.
        ##  - |shift_logits| = (batch_size, length - 1, n_vocabs)
        ##  - |shift_labels| = (batch_size, length - 1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        batch_size = shift_logits.size(0)
        output_size = shift_logits.size(-1)

        ## Flatten the tokens without reduction.
        ##  - |loss| = (batch_size,)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, output_size), shift_labels.view(-1)
        ).view(batch_size, -1)

        return loss

    # @torch.inference_mode()
    def ce_loss(self, gen_tokens: torch.Tensor) -> torch.Tensor:
        ##  - |gen_tokens| = (batch_size, length)
        ##  - |loss| = (batch_size,)
        loss = self._forward_without_reduction(
            input_ids=gen_tokens,
            labels=gen_tokens,
        ).mean(dim=-1)

        return loss

    @torch.inference_mode()
    def zlib_entropy(self, gen_tokens: torch.Tensor) -> torch.Tensor:
        ##  - |gen_tokens| = (batch_size, length)
        ##  - |zlib_entropy| = (batch_size,)
        zlib_entropy = [
            len(zlib.compress(bytes(sent, encoding="utf-8")))
            for sent in self.tok.batch_decode(
                gen_tokens, skip_special_tokens=True
            )
        ]
        ## Dtype: long -> float
        zlib_entropy = torch.FloatTensor(zlib_entropy)

        return zlib_entropy

    @torch.inference_mode()
    def zlib_entropy_ratio(self, gen_tokens: torch.Tensor) -> torch.Tensor:
        ##  - |gen_tokens| = (batch_size, length)
        ##  - |zlib_entropy| = (batch_size,)
        zlib_entropy_ratio = [
            len(zlib.compress(bytes(sent, encoding="utf-8")))
            / len(bytes(sent, encoding="utf-8"))
            for sent in self.tok.batch_decode(
                gen_tokens, skip_special_tokens=True
            )
        ]
        ## Dtype: long -> float
        zlib_entropy_ratio = torch.FloatTensor(zlib_entropy_ratio)

        return zlib_entropy_ratio

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
