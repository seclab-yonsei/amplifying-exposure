import torch
import lightning as L

import argparse

import multiprocessing as mp

from torch.utils.data import Dataset, DataLoader, IterableDataset

from typing import Dict


class MinimumRiskTrainingDataModule(L.LightningDataModule):
    def __init__(self, tok, config: argparse.Namespace):
        super().__init__()

        self.eos_token_id = tok.eos_token_id
        self.config = config

    def setup(self, stage: str):
        """Prepare a dataset.

        Args:
            stage (str): Meaningless arguments required for overwritting
        """
        # self.ds = EOSTokenDataset(
        #     eos_token_id=self.eos_token_id,
        #     samples_per_epoch=self.samples_per_epoch,
        # )
        self.ds = EOSTokenIterableDataset(eos_token_id=self.eos_token_id)

    def train_dataloader(self) -> DataLoader:
        """Returns a training dataloader.

        Returns:
            DataLoader: A training dataloader
        """
        return DataLoader(
            self.ds,
            batch_size=self.config.batch_size,
            num_workers=min(self.config.num_workers, mp.cpu_count()),
        )


class EOSTokenIterableDataset(IterableDataset):
    def __init__(self, eos_token_id: str = "[EOS]"):
        super(EOSTokenIterableDataset).__init__()

        self.eos_token_id = eos_token_id

    def __iter__(self) -> Dict[str, torch.Tensor]:
        """Returns a dictionary with eos_tokens as input_ids on every iteration.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing input_ids

        Yields:
            Iterator[Dict[str, torch.Tensor]]: A dictionary containing input_ids
        """
        ## Infinite iteration.
        while True:
            yield {"input_ids": torch.LongTensor([self.eos_token_id])}


class EOSTokenDataset(Dataset):
    def __init__(
        self,
        eos_token_id: str = "[EOS]",
        samples_per_epoch: int = 10_000,
    ):
        super(EOSTokenDataset).__init__()

        self.eos_token_id = eos_token_id
        self.len = samples_per_epoch

    def __len__(self) -> int:
        """Returns the total size of the dataset.

        Returns:
            int: The total size of the dataset
        """
        return self.len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns a dictionary with eos_tokens as input_ids.

        Args:
            idx (int): Index to reference in data

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing input_ids
        """
        return {"input_ids": torch.LongTensor([self.eos_token_id])}
