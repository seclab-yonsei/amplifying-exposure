import transformers
import pytorch_lightning as pl

import datetime
import easydict
import logging
import pprint
import yaml

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

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
    ).to(device="cuda", non_blocking=True)

    score_fn = GPT2Scorer(tok, model)

    ## Load a dataloader.
    tr_dataset = IterableDataset(tok)
    tr_dataloader = DataLoader(tr_dataset, batch_size=config.batch_size)

    ## Load a lightning module.
    lightning_module = MinimumRiskTrainingModule(model, tok, score_fn, config)

    ## Define a trainer.
    nowtime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_loggers = []
    if config.wandb_project:
        train_loggers.append(
            WandbLogger(name=nowtime, project=config.wandb_project)
        )

    trainer = pl.Trainer(
        logger=train_loggers,
        accelerator=config.accelerator,
        max_steps=config.max_steps,
        max_epochs=config.epochs,
        log_every_n_steps=config.logging_interval,
        precision=config.precision,
    )
    trainer.fit(lightning_module, tr_dataloader)


if __name__ == "__main__":
    config = define_config()
    main(config)
