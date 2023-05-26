import torch


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, tok):
        super(IterableDataset).__init__()
        self.tok = tok
        self.tokens = self.tok.encode(self.tok.eos_token, return_tensors="pt")

    def __iter__(self):
        ## Infinite iteration.
        while True:
            yield self.tokens
