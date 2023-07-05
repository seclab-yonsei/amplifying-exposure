import argparse
import logging
import pprint


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
