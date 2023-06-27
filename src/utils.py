import torch

import logging

from collections import OrderedDict


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


def remove_prefix_from_state_dict(
    state_dict: OrderedDict, prefix: str = "_forward_module.model."
) -> OrderedDict:
    new_state_dict = OrderedDict()
    for n, v in state_dict.items():
        name = n.replace(prefix, "")
        new_state_dict[name] = v

    return new_state_dict
