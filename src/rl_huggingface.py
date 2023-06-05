import torch
import transformers


class MinimumRiskTrainingTrainer(transformers.Trainer):
    def __init__(self, score_fn, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_fn = score_fn
        self.config = config

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

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer.
        By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        ## Maximum likelihood.
        x = inputs["input_ids"]
        # y = x.clone().detach().to(self.config.device)
        ## |x| = (batch_size, 1) -> only <EOS>

        prompt_len = x.size(1)
        assert prompt_len == 1

        ## Sampling y_hat.
        # gen_tokens = model.generate(

        gen_tokens = model.module.generate(
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
                # sampled_gen_tokens = model.generate(
                sampled_gen_tokens = model.module.generate(
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

        return loss
