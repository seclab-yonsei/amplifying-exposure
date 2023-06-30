import torch
import lightning as L
import transformers
import deepspeed

import argparse
import datetime
import logging

from lightning.pytorch.callbacks import (
    Callback,
    TQDMProgressBar,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.strategies import DeepSpeedStrategy
from pathlib import Path
from pytz import timezone
from typing import List

from src.dataset import MinimumRiskTrainingDataModule
from src.rl_lightning import MinimumRiskTrainingModule
from src.utils import define_logger, print_config


LOGGER = logging.getLogger(__name__)


def define_argparser() -> argparse.Namespace:
    """Defines the arguments that will be used to execute the code.

    Returns:
        argparse.Namespace: A dictionary whose arguments can be called
    """
    p = argparse.ArgumentParser()

    ## Deepspeed arguments (meaningless).
    p.add_argument("--local_rank", type=int, default=None)
    p.add_argument("--deepspeed", type=str, default=None)

    ## Model and tokenizer.
    p.add_argument(
        "--pretrained_model_name",
        type=str,
        default="EleutherAI/gpt-neo-1.3B",
        help=" ".join(
            [
                "Name of the model you want to fine-tune.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--revision",
        type=str,
        default="main",
        help=" ".join(
            [
                "Revision of the pretrained model.",
                "Default=%(default)s",
            ]
        ),
    )

    ## Dataset.
    p.add_argument(
        "--samples_per_epoch",
        type=int,
        default=10_000,
        help=" ".join(
            [
                "The number of training data to be included in one epoch.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help=" ".join(
            [
                "Number of samples to process in one training batch.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help=" ".join(
            [
                "Number of samples to process in one evaluation batch.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help=" ".join(
            [
                "How many subprocesses to use for data loading.",
                "0 means that the data will be loaded in the main process.",
                "Default=%(default)s",
            ]
        ),
    )

    ## Logger.
    p.add_argument(
        "--wandb_project",
        type=str,
        default="mrt",
        help=" ".join(
            [
                "Display name of the Wandb project.",
                "Default=%(default)s",
            ]
        ),
    )

    ## Callbacks.
    p.add_argument(
        "--ckpt",
        type=str,
        default="ckpt",
        help=" ".join(
            [
                "The location where the model checkpoint will be saved.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--every_n_epochs",
        type=int,
        default=1,
        help=" ".join(
            [
                "Number of epochs between checkpoints.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--save_top_k",
        type=int,
        default=-1,
        help=" ".join(
            [
                "The best k models according to the quantity monitored will be",
                "saved",
                "Default=%(default)s",
            ]
        ),
    )

    ## Trainer.
    p.add_argument(
        "--buffer_size",
        type=int,
        default=1_000,
        help=" ".join(
            [
                "The size of the buffer to store the pre-generated samples.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help=" ".join(
            [
                "Which hardware accelerator to use.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--devices",
        type=int,
        default=2,
        help=" ".join(
            [
                "Number of accelerators to use.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help=" ".join(
            [
                "Data types for model weights and biases.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=4,
        help=" ".join(
            [
                "Whether to update the gradient every few steps or not.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help=" ".join(
            [
                "Stop training once this number of epochs is reached.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--logging_interval",
        type=int,
        default=1,
        help=" ".join(
            [
                "How often to log within steps.",
                "Default=%(default)s",
            ]
        ),
    )

    ## Optimizer.
    p.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help=" ".join(
            [
                "The learning rate.",
                "Default=%(default)s",
            ]
        ),
    )

    ## Generation.
    p.add_argument(
        "--do_sample",
        action="store_true",
        help=" ".join(
            [
                "Whether or not to use sampling;",
                "use greedy decoding otherwise.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help=" ".join(
            [
                "Number of beams for beam search.",
                "1 means no beam search.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--min_new_tokens",
        type=int,
        default=64,
        help=" ".join(
            [
                "The minimum numbers of tokens to generate.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help=" ".join(
            [
                "The maximum numbers of tokens to generate.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help=" ".join(
            [
                "If set to int > 0, all ngrams of that size",
                "can only occur once.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=1,
        help=" ".join(
            [
                "If set to float < 1, only the smallest set of most probable",
                "tokens with probabilities that add up to top_p or higher are",
                "kept for generation.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=50,
        help=" ".join(
            [
                "The number of highest probability vocabulary tokens to keep",
                "for top-k-filtering.",
                "Default=%(default)s",
            ]
        ),
    )

    ## MRT.
    p.add_argument(
        "--rl_n_samples",
        type=int,
        default=1,
        help=" ".join(
            [
                "Number of samples to get baseline.",
                "Default=%(default)s",
            ]
        ),
    )

    ## Nowtime.
    kst = timezone("Asia/Seoul")
    nowtime = datetime.datetime.now(kst).strftime("%Y%m%d-%H%M%S")
    p.add_argument(
        "--nowtime",
        type=str,
        default=nowtime,
        help=" ".join(
            [
                "The time the learning script was run.",
                "Default=%(default)s",
            ]
        ),
    )

    ## Debug.
    p.add_argument(
        "--detect_anomaly",
        action="store_true",
        help=" ".join(
            [
                "Enable anomaly detection for the autograd engine.",
                "Default=%(default)s",
            ]
        ),
    )
    p.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help=" ".join(
            [
                "Specifies the debugging mode.",
                "Default=%(default)s",
            ]
        ),
    )

    config = p.parse_args()
    return config


def get_train_loggers(config: argparse.Namespace) -> Logger:
    """Returns the wandb logger to be used for training.

    Args:
        config (argparse.Namespace): The class that contains the configuration item

    Returns:
        Logger: A wandb logger (not list)
    """
    return WandbLogger(project=config.wandb_project, name=config.nowtime)


def get_callbacks(
    config: argparse.Namespace,
    refresh_rate: int = 1,
) -> List[Callback]:
    """Returns a list of callbacks.

    Args:
        config (argparse.Namespace): The class that contains the configuration item
        refresh_rate (int, optional): Determines at which rate (in number of batches) the progress bars get updated. Defaults to 1.

    Returns:
        List[Callback]: A list containing the callbacks
    """
    return [
        TQDMProgressBar(refresh_rate=refresh_rate),
        ModelCheckpoint(
            dirpath=str(Path(config.ckpt, config.nowtime)),
            verbose=True,
            every_n_epochs=config.every_n_epochs,
            save_top_k=config.save_top_k,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]


def main(config: argparse.Namespace) -> None:
    ## Print arguments.
    print_config(config)

    ## Logger.
    define_logger(config.debug)

    ## Force a build of cpu Adam in a Python shell.
    ## See: https://github.com/microsoft/DeepSpeed/issues/1846
    deepspeed.ops.op_builder.CPUAdamBuilder().load()

    ## See: https://sebastianraschka.com/blog/2023/llm-mixed-precision.html
    torch.set_float32_matmul_precision("medium")

    ## Auto-detect error.
    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    ## Load tokenizer and model.
    ## See: https://huggingface.co/EleutherAI/gpt-neo-2.7B
    tok = transformers.AutoTokenizer.from_pretrained(
        config.pretrained_model_name,
        revision=config.revision,
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
        logger=get_train_loggers(config),
        callbacks=get_callbacks(config),
        # fast_dev_run=True,
        accumulate_grad_batches=config.accumulate_grad_batches,
        max_epochs=config.max_epochs,
        log_every_n_steps=config.logging_interval,
        detect_anomaly=config.detect_anomaly,
        default_root_dir=config.ckpt,
    )

    ## And just train it.
    trainer.fit(lightning_module, datamodule=data_module)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
