import torch
import lightning as L

from torch.utils.data import DataLoader, IterableDataset


class MinimumRiskTrainingDataModule(L.LightningDataModule):
    def __init__(self, tok, config: dict):
        super().__init__()
        self.eos_token_id = tok.eos_token_id
        self.config = config
        
    def setup(self, stage: str):
        self.ds = IterableDataset(eos_token_id=self.eos_token_id)
        
    def train_dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )


class IterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, eos_token_id: str = "[EOS]"):
        super(IterableDataset).__init__()
        self.eos_token_id = eos_token_id

    def __iter__(self):
        ## Infinite iteration.
        while True:
            yield {"input_ids": torch.Tensor([self.eos_token_id]).long()}
