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
        self.alpha = 0.002

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

        ## Save hyper-parameters to self.hparams (auto-logged by W&B).
        self.save_hyperparameters(ignore=["model"])

    @torch.inference_mode()
    def on_train_start(self) -> None:
        ## Make a prompt.
        prompt = [self.tok.eos_token_id]
        input_ids = torch.LongTensor(prompt).to(self.model.device)

        prompt_len = input_ids.size(0)  ## not 1
        assert prompt_len == 1

        ## Prepare a local buffer (per device).
        buf = []

        ## Get ready to start `tqdm`.
        tqdm_iterator = tqdm.tqdm(
            desc=f"Preparing a replay buffer ({self.model.device})",
            position=(
                int(str(self.model.device).split(":")[-1])
                if str(self.model.device).startswith("cuda")
                else 0
            ),
            total=self.config.buffer_size,
        )
        with tqdm_iterator as pbar:
            for i in range(0, self.config.buffer_size, self.config.batch_size):
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
                assert gen_tokens.size(0) == rewards.size(0)
                buf += [
                    {"y": g, "reward": r} for g, r in zip(gen_tokens, rewards)
                ]

                ## Update pbar.
                pbar.update(gen_tokens.size(0))

        ## Extend our buffer.
        assert len(buf) == self.config.buffer_size
        self.replay_buffer = buf
        ## |replay_buffer| = (buffer_size,)

        ## Print information of the replay buffer.
        r = torch.stack([d["reward"] for d in buf])
        features = {
            "max": r.max(),
            "min": r.min(),
            "avg": r.mean(),
            "med": r.median(),
        }
        print(
            f"[Replay buffer ({self.model.device})]",
            ", ".join([f"{k}: {v:.2f}" for k, v in features.items()]),
        )

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
    def _get_reward(self, y: torch.Tensor) -> torch.Tensor:
        ## |y| = (batch_size, length)

        ## Calcualte reward.
        zlib = self.score_fn.zlib_entropy(y).to(y.device)
        logits = self(input_ids=y, return_dict=True).logits
        ppl = torch.exp(
            self.score_fn.ce_loss_without_reduction(logits=logits, labels=y)
        )
        ## |zlib| = (batch_size,)
        ## |logits| = (batch_size, length)
        ## |ppl| = (batch_size,)

        reward = zlib / ppl
        ## |reward| = (batch_size,)
        return reward

    @torch.inference_mode()
    def _fetch_reward_from_replay_buffer(self) -> tuple:
        ## Calculate the average.
        m = torch.stack([d["reward"] for d in self.replay_buffer]).mean()
        ## |m| = (1,)

        return m

    @torch.inference_mode()
    def _insert_y_hat_to_replay_buffer(
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
        ce_loss = self.score_fn.ce_loss_without_reduction(
            logits=logits, labels=y_hat
        )
        ## |y_hat| = (batch_size, length)
        ## |logits| = (batch_size, length, vocab_size)
        ## |ce_loss| = (batch_size,)

        with torch.no_grad():
            ## Fetch pseudy y.
            y_reward = self._fetch_reward_from_replay_buffer()
            y_reward.to(self.model.device)

            ## Based on the result of sampling, get reward.
            ## We set the generated sentence with the highest score
            ## in the batch as the target.
            actor_reward = self._get_reward(y_hat)
            ## |actor_reward| = (batch_size,)

            ## Take samples as many as n_samples,
            ## and get average rewards for them.
            # baseline = []

            # for _ in range(self.config.rl_n_samples):
            #     sampled_gen_tokens = self.model.generate(
            #         x,
            #         do_sample=self.config.do_sample,
            #         min_length=self.config.min_new_tokens + prompt_len,
            #         max_length=self.config.max_new_tokens + prompt_len,
            #         no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            #         top_p=self.config.top_p,
            #         top_k=self.config.top_k,
            #         synced_gpus=True,
            #     )
            #     baseline += [self._get_reward(sampled_gen_tokens)]

            # ## Get average of k samples.
            # baseline = torch.stack(baseline).mean(dim=0)
            ## |baseline| = (rl_n_samples, batch_size) -> (batch_size,)

            ## Now, we have relatively expected cumulative reward.
            ## Which score can be drawn from actor_reward substracted by baseline.
            reward = actor_reward - y_reward
            # reward = (actor_reward - baseline) / y_reward * 100.0
            ## |reward| = (batch_size,)

            ## Temp.
            zlib = actor_reward * torch.exp(ce_loss)

        ## Calculate minimum risk training loss.
        rl_loss = (ce_loss * -reward).mean()
        ce_loss = ce_loss.mean()

        loss = self.alpha * ce_loss + (1 - self.alpha) * rl_loss
        ## |loss| = (1,)

        ## Insert y_hat and their rewards.
        self._insert_y_hat_to_replay_buffer(
            y_hat.clone().detach(), actor_reward
        )

        ## Make a return dict.
        metrics = {
            "actor": actor_reward.mean(),
            "baseline": y_reward,
            # "baseline": baseline.mean(),
            "reward": reward.mean(),
            "zlib": zlib.mean(),
            "rl_loss": rl_loss,
            "ce_loss": ce_loss,
            "loss": loss,
        }
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True)

        return metrics
