import argparse
import logging

from typing import List

from src.utils import define_logger, print_config


LOGGER = logging.getLogger(__name__)


def define_argparser() -> argparse.Namespace:
    """Defines the arguments that will be used to execute the code.

    Returns:
        argparse.Namespace: A dictionary whose arguments can be called
    """
    p = argparse.ArgumentParser()

    config = p.parse_args()
    return config


def build_suffix_array(text: str) -> List[str]:
    ## Store each suffix together with its starting index.
    suffixes = [(text[i:], i) for i in range(len(text))]

    ## Sort suffixes alphabetically.
    suffixes.sort(key=lambda x: x[0])

    ## Store only the starting index of the sorted suffixes.
    suffix_array = [suffix[1] for suffix in suffixes]

    return suffix_array


def main(config: argparse.Namespace) -> None:
    ## Print arguments.
    print_config(config)

    ## Logger.
    define_logger(config.debug)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
