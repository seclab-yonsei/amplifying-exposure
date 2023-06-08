import torch
import lightning as L

from torch.utils.data import Dataset, DataLoader, IterableDataset


class MinimumRiskTrainingDataModule(L.LightningDataModule):
    def __init__(self, tok, config: dict):
        super().__init__()
        self.eos_token_id = tok.eos_token_id
        self.samples_per_epoch = config.samples_per_epoch
        self.config = config

    def setup(self, stage: str):
        self.ds = EOSTokenDataset(
            eos_token_id=self.eos_token_id,
            samples_per_epoch=self.samples_per_epoch,
        )
        # self.ds = EOSTokenIterableDataset(eos_token_id=self.eos_token_id)

    def train_dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )


class EOSTokenIterableDataset(IterableDataset):
    def __init__(self, eos_token_id: str = "[EOS]"):
        super(EOSTokenIterableDataset).__init__()
        self.eos_token_id = eos_token_id

    def __iter__(self):
        ## Infinite iteration.
        while True:
            yield {"input_ids": torch.Tensor([self.eos_token_id]).long()}


class EOSTokenDataset(Dataset):
    def __init__(
        self,
        eos_token_id: str = "[EOS]",
        samples_per_epoch: int = 1_000,
    ):
        super(EOSTokenDataset).__init__()
        self.eos_token_id = eos_token_id
        self.len = samples_per_epoch

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return {"input_ids": torch.Tensor([self.eos_token_id]).long()}
