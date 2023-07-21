import argparse
import json
import logging
import pathlib
import pprint

import pandas as pd

from pathlib import Path
from typing import List, Tuple, Dict, Union


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


def save_results(rslt: Union[List[dict], pd.DataFrame], save_path: str) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    ## Save pd.DataFrame to csv.
    if isinstance(rslt, pd.DataFrame):
        rslt = json.loads(rslt.to_json(orient="records"))

    ## Save.
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(rslt, f, indent=4)

    print(f"Results save to {save_path}")


def load_results(save_path: str) -> Tuple[List[dict], str]:
    ## Results.
    with open(save_path, "r", encoding="utf-8") as f:
        rslt = json.load(f)
    print(f"Results load from {save_path}")

    return rslt


# def load_results(
#     nowtime: str,
#     assets: str,
#     suffix: str = "",
# ) -> Tuple[List[dict], str]:
#     ## Find file names.
#     save_path = list(Path(assets).glob(f"*{nowtime}*{suffix}.json"))

#     assert len(save_path) == 1
#     save_path = save_path[0]

#     ## Results.
#     with open(save_path, "r", encoding="utf-8") as f:
#         rslt = json.load(f)
#     print(f"Results load from {save_path}")

#     return rslt, save_path


def calculate_similarity(
    token1: List[int],
    token2: List[int],
    n_gram: int = 3,
) -> bool:
    ## Calculate trigram similarity: str1 (reference) vs str2 (hyphothesis).
    ## It is same as "Is string 1 is similar with string 2?"

    ## Occasionally, if the Text consists of 1 or 2 words, the trigram multiset
    ## will result in an empty list, resulting in a divided by zero error.
    ## To prevent this, we guarantee that the trigram multiset has at least one element.

    ## We need to remove a EOS token.
    n_gram_set = lambda x: [
        " ".join(x[i : i + n_gram])
        for i in range(1, max(len(x) - n_gram, 1), 1)
    ]

    s1 = n_gram_set(token1)
    s2 = n_gram_set(token2)

    ## Return true if str1 is similar (or duplicated) to str2 else false.
    ## It is not recommended to mark two strings as similar, trivially.
    return len([i for i in s1 if i in s2]) / len(s1)
