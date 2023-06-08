import torch
import lightning as L

from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

import subprocess


class MinimumRiskTrainingModule(L.LightningModule):
    def __init__(self, tok, model, score_fn, config):
        super(MinimumRiskTrainingModule, self).__init__()

        self.tok = tok
        self.model = model
        self.score_fn = score_fn
        self.config = config

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.parameters(), lr=self.config.lr)

    def _get_reward(self, gen_tokens: torch.Tensor) -> dict:
        ## Calculate membership inference metrics for all samples.
        l = self.score_fn.ce_loss(gen_tokens)
        z = self.score_fn.zlib_entropy(gen_tokens).to(l.device)
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
            actor_reward = torch.max(r_dict["score"]) - r_dict["score"]
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
        loss = (r_dict["ce_loss"] * -reward).sum()

        ## Make a return dict.
        metrics = {"loss": loss}
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True)

        return metrics
