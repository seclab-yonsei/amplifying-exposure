import torch
import transformers
import lightning as L
import deepspeed

import datetime
import easydict
import logging
import pprint
import yaml

from lightning.pytorch.callbacks import (
    TQDMProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy
from pytz import timezone

from src.dataset import MinimumRiskTrainingDataModule
from src.rl_lightning import MinimumRiskTrainingModule
from src.score import GPTScorer
from src.utils import define_logger


LOGGER = logging.getLogger(__name__)


def define_config(fname: str = "config.yml") -> dict:
    ## Load yaml configuration file.
    with open(fname) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = easydict.EasyDict(config)

    if config.get("nowtime") == None:
        kst = timezone("Asia/Seoul")
        config.nowtime = datetime.datetime.now(kst).strftime("%Y%m%d-%H%M%S")

    return config


def get_train_loggers(wandb_project: str, nowtime: str):
    return WandbLogger(project=wandb_project, name=nowtime)


def get_callbacks(config: dict, refresh_rate: int = 1) -> list:
    return [
        TQDMProgressBar(refresh_rate=refresh_rate),
        ModelCheckpoint(
            dirpath=config.ckpt,
            verbose=True,
            every_n_epochs=config.every_n_epochs,
            save_top_k=config.save_top_k,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]


def main(config: dict) -> None:
    def print_config(config: dict) -> None:
        pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(config)

    print_config(config)

    ## Logger.
    define_logger(config)

    ## See:
    ##  - https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    ##  - https://sebastianraschka.com/blog/2023/llm-mixed-precision.html
    torch.set_float32_matmul_precision("medium")

    ## Auto-detect error.
    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    ## Load tokenizer and model.
    ## See: https://github.com/kakaobrain/kogpt
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
        torch_dtype="auto",  ## loaded as torch.float32 (not fp16)
        ## The argument 'low_cpu_mem_use=True'
        ## may cause RuntimeError: Tensors are not contiguous ...
        # low_cpu_mem_usage=True,
    )
    # returned_module = model.apply(make_weight_contiguous)

    ## Load a lightning module.
    lightning_module = MinimumRiskTrainingModule(tok, model, config)
    data_module = MinimumRiskTrainingDataModule(tok, config)

    ## Define a trainer.
    trainer = L.Trainer(
        accelerator=config.accelerator,
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            logging_batch_size_per_gpu=config.batch_size,
        ),
        devices=config.devices,
        precision=config.precision,
        logger=get_train_loggers(config.wandb_project, config.nowtime),
        callbacks=get_callbacks(config),
        # fast_dev_run=True,
        accumulate_grad_batches=config.accumulate_grad_batches,
        max_epochs=config.max_epochs,
        # max_steps=config.max_steps,
        log_every_n_steps=config.logging_interval,
        detect_anomaly=config.detect_anomaly,
        default_root_dir=config.ckpt,
    )

    ## And just train it.
    trainer.fit(lightning_module, datamodule=data_module)


if __name__ == "__main__":
    config = define_config()
    main(config)
