import torch

import logging


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
