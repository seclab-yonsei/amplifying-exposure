import torch
import transformers
import lightning as L

import datetime
import easydict
import logging
import os
import pprint
import yaml

from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy

from src.dataset import MinimumRiskTrainingDataModule
from src.rl_lightning import MinimumRiskTrainingModule
from src.score import GPTScorer
from src.utils import define_logger


LOGGER = logging.getLogger(__name__)

# .environ[
#     "HF_DATASETS_CACHE"
# ] = "/mnt/block-storage/.cache/huggingface/datasets"
# os.environ[os
#     "TRANSFORMERS_CACHE"
# ] = "/mnt/block-storage/.cache/huggingface/transformers"


def define_config(fname: str = "config.yml") -> dict:
    ## Load yaml configuration file.
    with open(fname) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = easydict.EasyDict(config)
    return config


def get_train_loggers(wandb_project: bool, nowtime: str):
    if wandb_project:
        return [WandbLogger(name=nowtime, project=wandb_project)]


def get_callbacks(refresh_rate: int = 1):
    return [TQDMProgressBar(refresh_rate=refresh_rate)]


def main(config: dict) -> None:
    def print_config(config: dict) -> None:
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(config)

    print_config(config)

    ## Logger.
    define_logger(config)

    ## Load tokenizer and model.
    tok = transformers.AutoTokenizer.from_pretrained(
        config.pretrained_model_name,
        revision=config.revision,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        mask_token="[MASK]",
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.pretrained_model_name,
        revision=config.revision,
        pad_token_id=tok.eos_token_id,
        torch_dtype=torch.float16,  ## not "auto"
        # low_cpu_mem_usage=True,
        ## The argument 'low_cpu_mem_use=True'
        ## may cause RuntimeError: Tensors are not contiguous ...
    )
    # returned_module = model.apply(make_weight_contiguous)

    score_fn = GPTScorer(tok, model)

    ## Load a lightning module.
    lightning_module = MinimumRiskTrainingModule(tok, model, score_fn, config)
    data_module = MinimumRiskTrainingDataModule(tok, config)

    ## Define a trainer.
    nowtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    trainer = L.Trainer(
        logger=get_train_loggers(config.wandb_project, nowtime),
        accelerator=config.accelerator,
        callbacks=get_callbacks(),
        devices=config.devices,
        # strategy=config.strategy,
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            logging_batch_size_per_gpu=config.batch_size,
        ),
        precision=config.precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        max_steps=config.max_steps,
        log_every_n_steps=config.logging_interval,
    )

    ## And just train it.
    trainer.fit(lightning_module, datamodule=data_module)


if __name__ == "__main__":
    config = define_config()
    main(config)
