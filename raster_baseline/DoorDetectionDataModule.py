from warnings import warn

import torch
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from dataset import DoorDataset

def _collate_fn(batch):
    return tuple(zip(*batch))

CLASSES = (
    "door",
)

class DoorDetectionDataModule(LightningDataModule):
    name = "doordetection"

    def __init__(
        self,
        data_dir: str,
        year: str = "2020",
        num_workers: int = 12,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.year = year
        self.data_dir = data_dir
        self.num_workers = num_workers

    @property
    def num_classes(self):
        return 1

    def train_dataloader(self, batch_size=1, transforms=None):
        """
        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        dataset = DoorDataset(root='../data/Public',
                              list='train_list_reduced_cnn.txt')
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_collate_fn,
        )
        return loader

    def val_dataloader(self, batch_size=1, transforms=None):
        """
        VOCDetection val set uses the `val` subset

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """

        dataset = DoorDataset(root='../data/Public',
                              list='valid_list_reduced_cnn.txt')
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=_collate_fn,
        )
        return loader
