import torch


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, eos_token_id: str = "[EOS]"):
        super(IterableDataset).__init__()
        self.eos_token_id = eos_token_id

    def __iter__(self):
        ## Infinite iteration.
        while True:
            yield torch.Tensor([self.eos_token_id]).long()
