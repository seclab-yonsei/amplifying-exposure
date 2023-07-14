import numpy as np

from typing import List, Dict


def make_pairs(scores: np.ndarray, texts: List[str]) -> List[Dict[str, str]]:
    ## Make indexes for pair them.
    idx = np.arange(len(scores))
    if len(idx) % 2:  ## make length to even
        idx = idx[:-1]

    ## Shuffle.
    np.random.shuffle(idx)
    scores = scores[idx]

    ## Make pairs.
    scores = scores.reshape(-1, 2)
    idx = idx.reshape(-1, 2)

    ## Determines chosen and rejected.
    pairs = []
    for (s1, s2), (i1, i2) in zip(scores, idx):
        ## A larger log probability is fake text
        ##  == Smaller negative log probability is fake text
        chosen_idx, rejected_idx = (i1, i2) if s1 < s2 else (i2, i1)

        chosen = texts[chosen_idx]
        rejected = texts[rejected_idx]

        items = {
            "prompt": "Human: " + "" + " Assistant:",  ## empty prompt
            "chosen": chosen,
            "rejected": rejected,
            "score_diff": float(abs(s1 - s2)),
        }
        pairs.append(items)

    return pairs
