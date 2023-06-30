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
        self.alpha = 0.002

        ## Auto-parameters.
        self.total_steps = int(
            np.ceil(self.config.max_steps / self.config.devices)
        )

        ## Replay buffer.
        self.replay_buffer = []

        ## Save hyper-parameters to self.hparams (auto-logged by W&B).
        self.save_hyperparameters(ignore=["model"])

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
            num_beams=self.config.num_beams,
            min_length=self.config.min_new_tokens + prompt_len,
            max_length=self.config.max_new_tokens + prompt_len,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
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

            ## Take samples as many as n_samples,
            ## and get average rewards for them.
            baseline = []

            for _ in range(self.config.rl_n_samples):
                sampled_y_hat: torch.Tensor = self.model.generate(
                    x,
                    do_sample=self.config.do_sample,
                    num_beams=self.config.num_beams,
                    min_length=self.config.min_new_tokens + prompt_len,
                    max_length=self.config.max_new_tokens + prompt_len,
                    no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                )
                baseline += [self._get_reward(sampled_y_hat)]

            ## Get average of k samples.
            ##  - |baseline| = (rl_n_samples, batch_size) -> (batch_size,)
            baseline = torch.stack(baseline).mean(dim=0)  ## not stack

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
        # self._insert_y_hat_to_replay_buffer(
        #     y_hat.clone().detach(), actor_reward
        # )

        ## Make a return dict.
        metrics = {
            "actor": actor_reward.mean(),
            "baseline": baseline.mean(),
            "reward": reward.mean(),
            "zlib": zlib.mean(),
            "rl_loss": rl_loss,
            "ce_loss": ce_loss,
            "loss": loss,
        }
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True)

        return metrics
