import torch
import torch.nn.functional as F
import lightning as L

import zlib

from deepspeed.ops.adam import DeepSpeedCPUAdam
from undecorated import undecorated
from types import MethodType


class MinimumRiskTrainingModule(L.LightningModule):
    def __init__(self, tok, model, config):
        super(MinimumRiskTrainingModule, self).__init__()

        self.tok = tok
        self.model = model
        self.config = config

        ## Force to calculate gradient graph.
        ## See: https://discuss.huggingface.co/t/how-to-output-loss-from-model-generate/16999
        # gen_with_grad = undecorated(self.model.generate)
        # self.model.generate_with_grad = MethodType(gen_with_grad, self.model)

        ## Save hyper-parameters to self.hparams (auto-logged by W&B).
        self.save_hyperparameters(ignore=["model"])

    def _get_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        reward: torch.Tensor = 1.0,
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
        loss = (
            loss_fct(shift_logits.view(-1, output_size), shift_labels.view(-1))
            .view(batch_size, -1)
            .mean(dim=-1)
        )

        reward = reward.to(device=loss.device, dtype=loss.dtype)
        loss = (loss * (1 + 1 / torch.sqrt(1 + reward))).sum()
        ## Following two equations are eventually same.
        ## \theta = \theta - risk * \nabla_\theta \log{P}
        ## \theta = \theta - -reward * \nabla_\theta \log{P}
        ## where risk = -reward.

        return loss

    @torch.inference_mode()
    def _get_reward(
        self, gen_tokens: torch.Tensor, alpha: float = 100.0
    ) -> torch.Tensor:
        ##  - |gen_tokens| = (batch_size, length)

        ## Calculate zlib entropy for all samples.
        ##  - |zlib_entropy| = (batch_size,)
        zlib_entropy = [
            len(zlib.compress(bytes(sent, encoding="utf-8")))
            for sent in self.tok.batch_decode(
                gen_tokens, skip_special_tokens=True
            )
        ]
        zlib_entropy = torch.FloatTensor(zlib_entropy) / alpha

        return zlib_entropy

    def forward(self, *args, **kwargs) -> dict:
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.parameters(), lr=self.config.lr)

    def training_step(self, batch: dict, batch_idx) -> torch.Tensor:
        ## Maximum likelihood.
        ##  - |x| = (batch_size, 1) -> only <EOS>
        x = batch["input_ids"]

        prompt_len = x.size(1)
        assert prompt_len == 1

        ## Sampling y_hat.
        ##  - |gen_tokens| = (batch_size, length)
        gen_tokens = self.model.generate(
            x,
            do_sample=True,
            temperature=self.config.temperature,  ## 1.0
            repetition_penalty=self.config.repetition_penalty,
            min_new_tokens=self.config.min_length - prompt_len,  ## 64 - 1
            max_new_tokens=self.config.max_length - prompt_len,  ## 64 - 1
            top_p=self.config.top_p,  ## 0.95
            top_k=self.config.top_k,  ## 40
        )

        with torch.no_grad():
            ## Based on the result of sampling, get reward.
            ## We set the generated sentence with the highest score
            ## in the batch as the target.
            ##  - |actor_reward| = (batch_size,)
            actor_reward = self._get_reward(gen_tokens)

            ## Take samples as many as n_samples, and get average rewards for them.
            ## I figured out that n_samples = 1 would be enough.
            baseline = []

            for _ in range(self.config.rl_n_samples):
                sampled_gen_tokens = self.model.generate(
                    x,
                    do_sample=True,
                    temperature=self.config.temperature,  ## 1.0
                    repetition_penalty=self.config.repetition_penalty,
                    min_new_tokens=self.config.min_length - prompt_len,  ## 64
                    max_new_tokens=self.config.max_length - prompt_len,  ## 64
                    top_p=self.config.top_p,  ## 0.95
                    top_k=self.config.top_k,  ## 40
                )
                baseline += [self._get_reward(sampled_gen_tokens)]

            ## Get average of k samples.
            ##  - |baseline| = (n_samples, batch_size) -> (batch_size,)
            baseline = torch.stack(baseline).mean(dim=0)

            ## Now, we have relatively expected cumulative reward.
            ## Which score can be drawn from actor_reward substracted by baseline.
            ##  - |reward| = (batch_size,)
            reward = F.relu(actor_reward - baseline)

        ## Calculate gradients with back-propagation.
        loss = self._get_loss(gen_tokens, gen_tokens, reward=reward)

        ## Make a return dict.
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True)

        return loss
