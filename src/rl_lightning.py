import torch
import lightning as L

import random
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
        """Fill the replay buffer before starting the training."""
        device = self.model.device

        ## Make a prompt.
        input_ids = torch.LongTensor([self.tok.eos_token_id]).to(device)

        prompt_len = input_ids.size(0)  ## not size(1)
        assert prompt_len == 1

        ## Prepare a local buffer (per device).
        buf = []

        ## Get ready to start `tqdm`.
        tqdm_iterator = tqdm.tqdm(
            desc=f"Preparing a replay buffer ({device})",
            position=(
                int(str(device).split(":")[-1])
                if str(device).startswith("cuda")
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
            f"[Replay buffer ({device})]",
            ", ".join([f"{k}: {v:.2f}" for k, v in features.items()]),
        )

    def forward(self, *args, **kwargs):
        """Model forward step.

        Returns:
            dict: Dictionary for return value
        """
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> dict:
        """Returns the configuration for the optimizers.

        Returns:
            dict: Configuration for optimizers
        """
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
        """Calculate the reward for the token you created.

        Args:
            y (torch.Tensor): Generated tokens (same as labels)
                - |y| = (batch_size, length)

        Returns:
            torch.Tensor: Per-element rewards
                - |reward| = (batch_size,)
        """
        ## Calcualte reward.
        zlib = self.score_fn.zlib_entropy(y).to(y.device)
        ## |zlib| = (batch_size,)

        logits = self(input_ids=y, return_dict=True).logits
        loss = self.score_fn.ce_loss_without_reduction(logits=logits, labels=y)
        ppl = torch.exp(loss)
        ## |logits| = (batch_size, length)
        ## |loss| = (batch_size,)
        ## |ppl| = (batch_size,)

        reward = zlib / ppl
        ## |reward| = (batch_size,)
        return reward

    @torch.inference_mode()
    def _fetch_reward_from_replay_buffer(self) -> torch.Tensor:
        """Calculate the average reward in the replay buffer.

        Returns:
            torch.Tensor: Average reward for replay buffer
                - |m| = (1,)
        """
        ## Calculate the average.
        m = torch.stack([d["reward"] for d in self.replay_buffer]).mean()
        ## |m| = (1,)
        return m

    @torch.inference_mode()
    def _insert_y_hat_to_replay_buffer(
        self,
        y_hat: torch.Tensor,
        y_hat_reward: torch.Tensor,
        order: str = "descending",
    ) -> None:
        """Add the y_hats generated by each step to the replay buffer.

        Args:
            y_hat (torch.Tensor): Generated tokens (same as labels)
            y_hat_reward (torch.Tensor): Reward for each generated token
        """
        ## Insert items.
        buf = [{"y": g, "reward": r} for g, r in zip(y_hat, y_hat_reward)]
        self.replay_buffer += buf

        ## Select items uniform randomly.
        assert order in ["ascending", "descending", "random"]
        if order == "ascending":
            self.replay_buffer = sorted(
                self.replay_buffer,
                key=itemgetter("reward"),
                reverse=False,
            )[: self.config.buffer_size]

        elif order == "descending":
            self.replay_buffer = sorted(
                self.replay_buffer,
                key=itemgetter("reward"),
                reverse=True,
            )[: self.config.buffer_size]

        elif order == "random":
            self.replay_buffer = random.sample(
                self.replay_buffer,
                k=self.config.buffer_size,
            )

    def training_step(self, batch: dict, batch_idx) -> dict:
        """Functions that overwrote the training step of the Lightning module.

        Args:
            batch (dict): A dictionary with input_ids as key
            batch_idx (_type_): An index for batch

        Returns:
            dict: A dictionary with loss
        """
        ## Fetch item.
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
            ## Based on the result of sampling, get reward.
            actor_reward = self._get_reward(y_hat)
            ## |actor_reward| = (batch_size,)

            ## We simplify the calculation process by calling the average reward
            ## of the sample, which we have already created with the replay buffer.
            baseline = self._fetch_reward_from_replay_buffer()
            baseline.to(self.model.device)
            ## |baseline| = (1,)

            ## Now, we have relatively expected cumulative reward.
            ## Which score can be drawn from actor_reward substracted by baseline.
            reward = actor_reward - baseline
            ## |reward| = (batch_size,)

            ## Compute zlib entropy separately for logging.
            zlib = actor_reward * torch.exp(ce_loss)
            ## |zlib| = (batch_size,)

        ## Calculate minimum risk training loss.
        rl_loss = (ce_loss * -reward).mean()
        ce_loss = ce_loss.mean()
        ## |rl_loss| = (1,)
        ## |ce_loss| = (1,)

        loss = self.alpha * ce_loss + (1 - self.alpha) * rl_loss
        ## |loss| = (1,)

        ## Insert y_hat and their rewards.
        self._insert_y_hat_to_replay_buffer(
            y_hat.clone().detach(), actor_reward
        )

        ## Make a return dict.
        metrics = {
            "actor": actor_reward.mean(),
            "baseline": baseline,
            "reward": reward.mean(),
            "zlib": zlib.mean(),
            "rl_loss": rl_loss,
            "ce_loss": ce_loss,
            "loss": loss,
        }
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True)

        return metrics
