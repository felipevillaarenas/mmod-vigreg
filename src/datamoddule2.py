import itertools
import os

import pytorch_lightning as pl
import pytorchvideo.data
import torch

from torchvision.datasets import Kinetics
from torch.utils.data import DataLoader


from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torch.utils.data import DistributedSampler, RandomSampler
from torchaudio.transforms import MelSpectrogram, Resample
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)

class KineticsDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_dir: str = "./data/kinetics400/s",
        batch_size: int = 32,
        workers: int = 0,
        num_classes="400",
        transform = None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.workers = workers
        self.num_classes = num_classes
        self.transform = transform

    def prepare_data(self):
        # download
        Kinetics(self.data_dir, frames_per_clip=1, split="val", num_classes=self.num_classes,download=True)

    def setup(self):


        self.val = Kinetics(
            self.data_dir,
            split=False,
            num_classes=self.num_classes,
            transform=self.transform
            )


    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.worker
            )


if __name__ == "__main__":
    dm = KineticsDataModule()
    print("run exp")

