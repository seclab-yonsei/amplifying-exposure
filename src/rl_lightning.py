import torch
import torch.nn.functional as F
import lightning as L

import easydict
import zlib

import numpy as np

from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.accelerator import get_accelerator
from undecorated import undecorated
from types import MethodType

from src.score import GPTScorer


class MinimumRiskTrainingModule(L.LightningModule):
    def __init__(self, tok, model, config):
        super(MinimumRiskTrainingModule, self).__init__()

        self.tok = tok
        self.model = model
        self.config = config

        self.score_fn = GPTScorer(tok=tok, model=model)
        self.alpha = 30.0

        ## Force to calculate gradient graph.
        ## See: https://discuss.huggingface.co/t/how-to-output-loss-from-model-generate/16999
        gen_with_grad = undecorated(self.model.generate)
        self.model.generate_with_grad = MethodType(gen_with_grad, self.model)

        ## Save hyper-parameters to self.hparams (auto-logged by W&B).
        self.save_hyperparameters(ignore=["model"])

    # def _get_weighted_loss(
    #     self,
    #     loss: torch.Tensor,
    #     reward: torch.Tensor,
    # ) -> torch.Tensor:
    #     ##  - |loss| = (batch_size,)
    #     ##  - |reward| = (batch_size,)
    #     reward = reward.to(device=loss.device, dtype=loss.dtype)
    #     loss = (loss * (-reward * 10 + 1)).mean()

    #     # loss = (loss * (1 + 1 / torch.sqrt(1 + reward))).mean()

    #     # loss = (loss * torch.exp(self.alpha / reward)).mean()
    #     # loss = (loss * (1 + F.sigmoid(-reward / self.alpha))).sum()
    #     # loss = (loss * (1 + F.sigmoid(-reward / self.alpha))).sum()
    #     # loss = (loss * F.relu6(1 / reward)).sum()
    #     # loss = (loss * (1 + 1 / torch.sqrt(1 + reward / self.alpha))).mean()

    #     ## Following two equations are eventually same.
    #     ## \theta = \theta - risk * \nabla_\theta \log{P}
    #     ## \theta = \theta - -reward * \nabla_\theta \log{P}
    #     ## where risk = -reward.

    #     return loss

    def _get_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, reward: torch.Tensor
    ) -> torch.Tensor:
        ## |logits| = (batch_size, length, n_vocabs)
        ## |labels| = (batch_size, length,)
        ## |reward| = (batch_size,)

        ## Move labels to correct device to enable model parallelism.
        labels = labels.to(logits.device)
        reward = reward.to(logits.device)

        ## Shift so that tokens < n predict n.
        ##  - |shift_logits| = (batch_size, length - 1, n_vocabs)
        ##  - |shift_labels| = (batch_size, length - 1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        ## Flatten the tokens without reduction.
        ##  - |ce_loss| = (batch_size,)
        ce_loss = (
            F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.tok.pad_token_id,
                reduction="none",
            )
            .view(labels.size(0), -1)
            .sum(-1)
        )
        ce_loss = (ce_loss * -reward).mean()

        return ce_loss

    def _get_reward(self, tokens: torch.Tensor) -> dict:
        zlib_entropy = [
            len(zlib.compress(bytes(sent, encoding="utf-8")))
            for sent in self.tok.batch_decode(tokens, skip_special_tokens=True)
        ]
        ## Dtype: long -> float
        reward = torch.FloatTensor(zlib_entropy) / 100

        return reward

        # ## Calculate membership inference metrics for all samples.
        # l = self.score_fn.ce_loss(gen_tokens)
        # p = torch.exp(l)
        # z = self.score_fn.zlib_entropy(gen_tokens).to(l.device)
        # ## |l| = (batch_size,)
        # ## |p| = (batch_size,)
        # ## |z| = (batch_size,)

        # reward = z
        # ## |reward| = (batch_size,)

        # return easydict.EasyDict(
        #     {"ce": l, "ppl": p, "zlib": z, "reward": reward}
        # )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.parameters(), lr=self.config.lr)

    def training_step(self, batch: dict, batch_idx) -> torch.Tensor:
        ## Flush the cache before start a training loop.
        get_accelerator().empty_cache()

        ## Maximum likelihood.
        ##  - |x| = (batch_size, 1) -> only <EOS>
        x = batch["input_ids"]

        prompt_len = x.size(1)
        assert prompt_len == 1

        ## Sampling y_hat.
        ##  - |gen_tokens| = (batch_size, length)
        gen_tokens = self.model.generate(
            x,
            do_sample=self.config.do_sample,
            # temperature=self.config.temperature,  ## 1.0
            # repetition_penalty=self.config.repetition_penalty,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            # min_new_tokens=self.config.min_new_tokens,  ## 64
            max_new_tokens=self.config.max_new_tokens,  ## 64
            top_p=self.config.top_p,  ## 0.95
            top_k=self.config.top_k,  ## 40
        )

        with torch.no_grad():
            ## Based on the result of sampling, get reward.
            ## We set the generated sentence with the highest score
            ## in the batch as the target.
            ##  - |actor_reward| = (batch_size,)
            actor_reward = self._get_reward(gen_tokens)

            ## Take samples as many as n_samples,
            ## and get average rewards for them.
            baseline = []

            for _ in range(self.config.rl_n_samples):
                sampled_gen_tokens = self.model.generate(
                    x,
                    do_sample=self.config.do_sample,
                    # temperature=self.config.temperature,  ## 1.0
                    # repetition_penalty=self.config.repetition_penalty,
                    no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                    # min_new_tokens=self.config.min_new_tokens,  ## 64
                    max_new_tokens=self.config.max_new_tokens,  ## 64
                    top_p=self.config.top_p,  ## 0.95
                    top_k=self.config.top_k,  ## 40
                )
                baseline += [self._get_reward(sampled_gen_tokens)]

            ## Get average of k samples.
            ##  - |baseline| = (rl_n_samples, batch_size) -> (batch_size,)
            baseline = torch.stack(baseline).mean(dim=0)  ## not stack

            ## Now, we have relatively expected cumulative reward.
            ## Which score can be drawn from actor_reward substracted by baseline.
            ##  - |reward| = (batch_size,) \in (-inf, +inf)
            reward = actor_reward - baseline

        ## Forward.
        logits = self(input_ids=gen_tokens, return_dict=True).logits
        labels = gen_tokens

        loss = self._get_loss(logits, labels, reward=reward)

        # loss = self._get_weighted_loss(r_dict.ce, reward=reward)

        ## Make a return dict.
        metrics = {
            "actor": actor_reward.mean(),
            "baseline": baseline.mean(),
            "reward": reward.mean(),
            "loss": loss,
        }

        # ## Make a return dict.
        # metrics = {
        #     # "actor":
        #     "loss": loss,
        #     "loss_mle": loss_mle.mean(),
        #     "loss_gmrt": loss_gmrt.mean(),
        #     "ce": r_dict.ce.mean(),
        #     "ppl": r_dict.ppl.mean(),
        #     "zlib_entropy": r_dict.zlib.mean(),
        #     # "baseline": baseline,
        #     # "reward": reward,
        # }
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True)

        return metrics
