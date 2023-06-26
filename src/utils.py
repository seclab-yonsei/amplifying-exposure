import torch

import logging
import shutil

from lightning.pytorch.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

from pathlib import Path


def define_logger(debug: bool = False) -> None:
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    level = logging.DEBUG if debug else logging.INFO

    ## Save log.
    logging.basicConfig(level=level, format=log_format)


def make_weights_contiguous(module: torch.nn.Module):
    ## Ref: https://jh-bk.tistory.com/10
    for p in module.parameters():
        if not p.is_contiguous():
            p.data = p.data.contiguous()


def convert_ckpt_as_one_file(
    checkpoint_dir: str, make_clean: bool = True
) -> None:
    ## Auto-naming.
    output_file = Path(
        Path(checkpoint_dir).parent,
        "{}.pt".format(Path(checkpoint_dir).name.split(".")[0]),
    )

    ## Convert.
    convert_zero_checkpoint_to_fp32_state_dict(
        checkpoint_dir=checkpoint_dir, output_file=output_file
    )

    ## Make clean.
    if make_clean:
        shutil.rmtree(checkpoint_dir)
        print(f"Checkpoint folder '{checkpoint_dir}' is now clean")
