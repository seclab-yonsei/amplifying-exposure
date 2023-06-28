import torch

import argparse
import logging
import pprint

from collections import OrderedDict


def define_logger(debug: bool = False) -> None:
    """Define a Logger.

    Args:
        debug (bool, optional): Specifies the debugging mode. Defaults to False.
    """
    log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    level = logging.DEBUG if debug else logging.INFO

    ## Save log.
    logging.basicConfig(level=level, format=log_format)


def print_config(config: argparse.Namespace) -> None:
    """Display configuration items beautifully.

    Args:
        config (argparse.Namespace): The class that contains the configuration item
    """
    pprint.PrettyPrinter(indent=4, sort_dicts=False).pprint(vars(config))


def make_weights_contiguous(module: torch.nn.Module):
    ## Ref: https://jh-bk.tistory.com/10
    for p in module.parameters():
        if not p.is_contiguous():
            p.data = p.data.contiguous()


def remove_prefix_from_state_dict(
    state_dict: OrderedDict,
    prefix: str = "_forward_module.model.",
) -> OrderedDict:
    new_state_dict = OrderedDict()
    for n, v in state_dict.items():
        name = n.replace(prefix, "")
        new_state_dict[name] = v

    return new_state_dict
