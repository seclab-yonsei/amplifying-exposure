import torch
import transformers
import lightning as L
import deepspeed

import datetime
import easydict
import logging
import pprint
import yaml

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy

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
    return config


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
        low_cpu_mem_usage=True,
    )
    score_fn = GPTScorer(tok, model)

    ## Make model weights to be contigous
    for p in model.parameters():
        assert p.is_contiguous(), p
    # assert all([p.is_contiguous() for p in model.parameters()]), "Error!"

    ## Load a lightning module.
    lightning_module = MinimumRiskTrainingModule(tok, model, score_fn, config)
    data_module = MinimumRiskTrainingDataModule(tok, config)

    ## Define a trainer.
    train_loggers = []
    if config.wandb_project:
        nowtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_loggers.append(
            WandbLogger(name=nowtime, project=config.wandb_project)
        )

    trainer = L.Trainer(
        logger=train_loggers,
        accelerator=config.accelerator,
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
