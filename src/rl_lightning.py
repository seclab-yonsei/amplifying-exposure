import torch
import lightning as L

import numpy as np

from deepspeed.ops.adam import DeepSpeedCPUAdam

from src.score import GPTScorer


class MinimumRiskTrainingModule(L.LightningModule):
    def __init__(self, tok, model, config):
        super(MinimumRiskTrainingModule, self).__init__()

        self.tok = tok
        self.model = model
        self.config = config

        self.score_fn = GPTScorer(tok=tok)
        self.total_batch_size = (
            self.config.devices
            * self.config.batch_size
            * self.config.accumulate_grad_batches
        )
        self.total_steps = int(
            np.ceil(
                self.config.samples_per_epoch
                * self.config.max_epochs
                / self.total_batch_size
            )
        )
        self.alpha = 0.017
        self.replay_buffer = []

        ## Save hyper-parameters to self.hparams (auto-logged by W&B).
        self.save_hyperparameters(ignore=["model"])

    @torch.inference_mode()
    def _get_reward(self, y: torch.Tensor) -> torch.Tensor:
        ## |y| = (batch_size, length)

        ## Calcualte reward.
        zlib = self.score_fn.zlib_entropy(y).to(y.device)
        logits = self(input_ids=y, return_dict=True).logits
        ## |zlib| = (batch_size,)
        ## |logits| = (batch_size, length)

        reward = zlib / self.score_fn._ce_loss_without_reduction(
            logits=logits, labels=y
        )
        ## |reward| = (batch_size,)
        return reward

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.lr,
            total_steps=self.total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "Learning Rate",
            },
        }

    """
    def on_train_epoch_start(self):
        input_ids = (
            torch.tensor([self.tok.eos_token_id], dtype=torch.int32)
            .repeat(self.config.batch_size)
            .to(self.model.device)
        )
        prompt_len = input_ids.size(1)

        buffer = []
        n_iter = int(
            np.ceil(
                self.config.buffer_size
                / (self.config.batch_size * self.config.devices)
            )
        )
        for i in tqdm.tqdm(range(n_iter), desc="Preparing the replay buffer"):
            bs = i * self.config.batch_size
            gen_tokens = (
                self.model.generate(
                    input_ids,
                    do_sample=self.config.do_sample,
                    min_length=self.config.min_new_tokens + prompt_len,
                    max_length=self.config.max_new_tokens + prompt_len,
                    no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    synced_gpus=True,
                )
                .clone()
                .detach()
            )
            rewards = self._get_reward(gen_tokens)
            for ii in range(gen_tokens.size(0)):
                buffer.append(
                    {
                        "y": gen_tokens[ii],
                        "reward": rewards[ii],
                    }
                )

        ## Extend our buffer.
        self.replay_buffer += buffer
    """

    def training_step(self, batch: dict, batch_idx) -> dict:
        ## Maximum likelihood.
        x = batch["input_ids"]
        ## |x| = (batch_size, 1) (i.e., only [<EOS>,])

        prompt_len = x.size(1)
        assert prompt_len == 1

        ## Sample y_hat, logits, and the loss.
        y_hat: torch.Tensor = self.model.generate(
            x,
            do_sample=self.config.do_sample,
            min_length=self.config.min_new_tokens + prompt_len,
            max_length=self.config.max_new_tokens + prompt_len,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            synced_gpus=True,
        )
        logits: torch.Tensor = self(input_ids=y_hat, return_dict=True).logits
        ce_loss = self.score_fn._ce_loss_without_reduction(
            logits=logits, labels=y_hat
        )
        ## |y_hat| = (batch_size, length)
        ## |logits| = (batch_size, length, vocab_size)
        ## |ce_loss| = (batch_size,)

        with torch.no_grad():
            ## Based on the result of sampling, get reward.
            ## We set the generated sentence with the highest score
            ## in the batch as the target.
            actor_reward = self._get_reward(y_hat)
            ## |actor_reward| = (batch_size,)

            ## Take samples as many as n_samples,
            ## and get average rewards for them.
            baseline = []

            for _ in range(self.config.rl_n_samples):
                sampled_gen_tokens = self.model.generate(
                    x,
                    do_sample=self.config.do_sample,
                    min_length=self.config.min_new_tokens + prompt_len,
                    max_length=self.config.max_new_tokens + prompt_len,
                    no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    synced_gpus=True,
                )
                baseline += [self._get_reward(sampled_gen_tokens)]

            ## Get average of k samples.
            baseline = torch.stack(baseline).mean(dim=0)
            ## |baseline| = (rl_n_samples, batch_size) -> (batch_size,)

            ## Now, we have relatively expected cumulative reward.
            ## Which score can be drawn from actor_reward substracted by baseline.
            reward = actor_reward - baseline
            # reward = (actor_reward - baseline) / torch.max(actor_reward) * 100.0
            ## |reward| = (batch_size,)

        ## Calculate minimum risk training loss.
        rl_loss = (ce_loss * -reward).mean()
        ce_loss = ce_loss.mean()

        loss = self.alpha * ce_loss + (1 - self.alpha) * rl_loss  ## alpha=0.1
        ## |loss| = (1,)

        ## Make a return dict.
        metrics = {
            "actor": actor_reward.mean(),
            "baseline": baseline.mean(),
            "reward": reward.mean(),
            "rl_loss": rl_loss,
            "ce_loss": ce_loss,
            "loss": loss,
        }
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True)

        return metrics
