import torch
import transformers
import lightning as L
import deepspeed

import datetime
import easydict
import logging
import pprint
import yaml

from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger

from src.dataset import IterableDataset
from src.rl_module import MinimumRiskTrainingModule
from src.score import GPT2Scorer
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

    ## Initialize the distributed training strategy.
    # torch.distributed.init_process_group(backend="nccl", init_method="env://")
    deepspeed.init_distributed()

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
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

    score_fn = GPT2Scorer(tok, model)

    ## Load a dataloader.
    tr_dataset = IterableDataset(eos_token_id=tok.eos_token_id)
    tr_dataloader = DataLoader(
        tr_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    ## Load a lightning module.
    lightning_module = MinimumRiskTrainingModule(model, tok, score_fn, config)

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
        strategy="fsdp",
        precision=config.precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        max_steps=config.max_steps,
        log_every_n_steps=config.logging_interval,
    )

    ## And just train it.
    trainer.fit(lightning_module, tr_dataloader)


if __name__ == "__main__":
    config = define_config()
    main(config)
