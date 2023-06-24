import torch
import lightning as L

import tqdm

import numpy as np

from deepspeed.ops.adam import DeepSpeedCPUAdam
from operator import itemgetter

from src.score import GPTScorer


class MinimumRiskTrainingModule(L.LightningModule):
    def __init__(self, tok, model, config):
        super(MinimumRiskTrainingModule, self).__init__()

        self.tok = tok
        self.model = model
        self.config = config

        self.score_fn = GPTScorer(tok=tok)
        self.alpha = 0.017

        ## Auto-parameters.
        self.per_replica_batch_size = (
            self.config.batch_size * self.config.accumulate_grad_batches
        )
        self.global_batch_size = (
            self.per_replica_batch_size * self.config.devices
        )
        self.total_steps = (
            int(np.ceil(self.config.samples_per_epoch / self.global_batch_size))
            * self.config.max_epochs
        )
        self.replay_buffer = None  ## per device

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

    def configure_optimizers(self) -> dict:
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

    @torch.inference_mode()
    def on_train_epoch_start(self) -> None:
        ## Make a prompt.
        prompt = [self.tok.eos_token_id]
        input_ids = torch.LongTensor(prompt).to(self.model.device)

        prompt_len = input_ids.size(0)  ## not 1
        assert prompt_len == 1

        ## Prepare a local buffer (per device).
        buf = []

        ## Get ready to start `tqdm`.
        iterable = range(0, self.config.buffer_size, self.config.batch_size)
        desc = f"Preparing a replay buffer ({self.model.device})"
        position = (
            int(str(self.model.device).split(":")[-1])
            if str(self.model.device).startswith("cuda")
            else 0
        )
        for i in tqdm.tqdm(iterable, desc=desc, position=position):
            bs = (  ## more batch size can be allowed in inference mode
                self.config.batch_size
                if i + self.config.batch_size < self.config.buffer_size
                else self.config.buffer_size - i
            )
            gen_tokens = (
                self.model.generate(
                    input_ids.repeat(bs, 1),
                    do_sample=self.config.do_sample,
                    min_length=self.config.min_new_tokens + prompt_len,
                    max_length=self.config.max_new_tokens + prompt_len,
                    no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    # synced_gpus=True,
                )
                .clone()
                .detach()
            )
            rewards = self._get_reward(gen_tokens)

            ## Stack.
            buf += [{"y": g, "reward": r} for g, r in zip(gen_tokens, rewards)]

        ## Extend our buffer.
        assert self.replay_buffer == None
        self.replay_buffer = buf
        assert (
            len(self.replay_buffer) == self.config.buffer_size
        ), f"{len(self.replay_buffer)} != {self.config.buffer_size}"

    @torch.inference_mode()
    def fetch_y_from_replay_buffer(self) -> tuple:
        ## Sort buffer by ascending order.
        self.replay_buffer = sorted(
            self.replay_buffer, key=itemgetter("reward")
        )
        element = self.replay_buffer[-1]

        return (element["y"], element["reward"])

    @torch.inference_mode()
    def insert_y_hat_to_replay_buffer(
        self, y_hat: torch.Tensor, y_hat_reward: torch.Tensor
    ) -> None:
        ## Insert items.
        buf = [{"y": g, "reward": r} for g, r in zip(y_hat, y_hat_reward)]
        self.replay_buffer += buf

        ## Select top-k items.
        self.replay_buffer = sorted(
            self.replay_buffer, key=itemgetter("reward"), reverse=True
        )[: self.config.buffer_size]

    def training_step(self, batch: dict, batch_idx) -> dict:
        ## Maximum likelihood.
        x = batch["input_ids"]
        ## |x| = (batch_size, 1) (i.e., only [<EOS>,])

        prompt_len = x.size(1)
        assert prompt_len == 1

        ## Fetch pseudy y.
        y, y_reward = self.fetch_y_from_replay_buffer()
        y_reward.to(self.model.device)

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
            reward = (actor_reward - baseline) / y_reward * 100.0
            ## |reward| = (batch_size,)

        ## Calculate minimum risk training loss.
        rl_loss = (ce_loss * -reward).mean()
        ce_loss = ce_loss.mean()

        loss = self.alpha * ce_loss + (1 - self.alpha) * rl_loss  ## alpha=0.1
        ## |loss| = (1,)

        ## Insert y_hat and their rewards.
        self.insert_y_hat_to_replay_buffer(y_hat.clone().detach(), actor_reward)

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
