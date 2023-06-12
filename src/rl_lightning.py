import torch
import torch.nn.functional as F
import lightning as L

import zlib

import numpy as np

from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class MinimumRiskTrainingModule(L.LightningModule):
    def __init__(self, tok, model, config):
        super(MinimumRiskTrainingModule, self).__init__()

        self.tok = tok
        self.model = model
        self.config = config

        ## Save hyper-parameters to self.hparams (auto-logged by W&B).
        self.save_hyperparameters(ignore=["model"])

    def _forward_without_reduction(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        ## |input_ids| = (batch_size, length)
        ## |labels| = (batch_size, length)

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
        loss = (
            loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            .view(-1, shift_labels.size(1))
            .mean(dim=1)
        )
        print(loss, loss.size())

        ## Return.
        return loss

    def ce_loss(self, batch: torch.Tensor) -> torch.Tensor:
        ## Calcualte perplexity.
        # input_ids = batch.clone()
        # labels = torch.cat([batch.clone()[:, 1:], batch.clone()[:, :1]], dim=1)

        ## Forward and average it by token dimension.
        loss = self._forward_without_reduction(
            input_ids=batch,
            labels=batch,
            # labels=torch.cat([batch[:, 1:], batch[:, :1]], dim=1),
            # input_ids=input_ids, labels=labels
        )
        loss = loss.mean(dim=-1)

        return loss

    def zlib_entropy(self, batch: torch.Tensor) -> torch.Tensor:
        ## Calculate zlib entropy.
        ##  - |entropy| = (batch_size,)
        entropy = [
            len(zlib.compress(bytes(s, encoding="utf-8")))
            for s in self.tok.batch_decode(batch, skip_special_tokens=True)
        ]
        entropy = torch.tensor(entropy, dtype=batch.dtype).contiguous()

        return entropy

    # def window_perplexity(
    #     self,
    #     batch: np.ndarray,
    #     window_size: int = 50,
    #     stride: int = 8,
    # ) -> np.ndarray:
    #     ppls = []
    #     for sp in range(0, batch.shape[1] - window_size, stride):
    #         ppl = self.perplexity(batch[:, sp : sp + window_size])
    #         ppls.append(ppl)

    #     ## Calculate the minimum ppl.
    #     ##  - |ppl| = (batch_size,)
    #     ppl = np.min(np.stack(ppls), axis=0)

    #     return ppl

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.parameters(), lr=self.config.lr)

    def _get_reward(self, gen_tokens: torch.Tensor) -> dict:
        ## Calculate membership inference metrics for all samples.

        ## Since zlib entropy is too large (e.g., 600, ...), it may cause
        ## nan value error like:
        ## >>> RuntimeError: Function 'NativeDropoutBackward0' returned nan values in its 0th output.

        # l = self.score_fn.ce_loss(gen_tokens)
        # z = self.score_fn.zlib_entropy(gen_tokens).to(l.device)

        l = self.ce_loss(gen_tokens)
        z = self.zlib_entropy(gen_tokens).to(l.device)
        # w = self.score_fn.window_perplexity(gen_tokens)
        ## |l| = (batch_size,)
        ## |z| = (batch_size,)

        s = z / l  ## score
        ## |s| = (batch_size,)

        return {"ce_loss": l, "zlib": z, "score": s}

    def training_step(self, batch, batch_idx):
        ## Maximum likelihood.
        x = batch["input_ids"]
        # y = x.clone().detach().to(self.config.device)
        ## |x| = (batch_size, 1) -> only <EOS>

        prompt_len = x.size(1)
        assert prompt_len == 1

        ## Sampling y_hat.
        gen_tokens = self.model.generate(
            x,
            do_sample=True,
            temperature=self.config.temperature,  ## 1.0
            repetition_penalty=self.config.repetition_penalty,
            min_new_tokens=self.config.min_length - prompt_len,  ## 256 - 1
            max_new_tokens=self.config.max_length - prompt_len,  ## 256 - 1
            top_p=self.config.top_p,  ## 0.95
            top_k=self.config.top_k,  ## 40
        )
        ## |gen_tokens| = (batch_size, length)

        r_dict = self._get_reward(gen_tokens)

        with torch.no_grad():
            ## Based on the result of sampling, get reward.
            ## We set the generated sentence with the highest score
            ## in the batch as the target.
            actor_reward = (
                torch.max(r_dict["score"]) - r_dict["score"]
            )  ## usually zero
            # actor_reward = r_dict["score"]  ## going higher

            ## |y_hat| = (batch_size, length, output_size)
            ## |indices| = (batch_size, length)
            ## |actor_reward| = (batch_size)

            ## Take samples as many as n_samples, and get average rewards for them.
            ## I figured out that n_samples = 1 would be enough.
            baseline = []

            for _ in range(self.config.rl_n_samples):
                sampled_gen_tokens = self.model.generate(
                    x,
                    do_sample=True,
                    temperature=self.config.temperature,  ## 1.0
                    repetition_penalty=self.config.repetition_penalty,
                    min_new_tokens=self.config.min_length - prompt_len,  ## 256
                    max_new_tokens=self.config.max_length - prompt_len,  ## 256
                    top_p=self.config.top_p,  ## 0.95
                    top_k=self.config.top_k,  ## 40
                )
                baseline += [self._get_reward(sampled_gen_tokens)["score"]]

            ## Get average of k samples.
            baseline = torch.stack(baseline).mean(dim=0)
            ## |baseline| = (n_samples, batch_size) -> (batch_size,)

            ## Now, we have relatively expected cumulative reward.
            ## Which score can be drawn from actor_reward substracted by baseline.
            reward = actor_reward - baseline
            ## |reward| = (batch_size,)

        ## Calculate gradients with back-propagation.
        # loss = (self.score_fn.ce_loss(gen_tokens) * -reward).sum()
        loss = (r_dict["ce_loss"] * -reward).sum()

        ## Make a return dict.
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True)

        return loss
