import itertools

import pytorch_lightning
import torch

import datamodule_map.datasets as datasets

from datamodule_map.transforms import MultiModeTrainDataTransform


class KineticsDataModule(pytorch_lightning.LightningDataModule):
    """
    This LightningDataModule implementation constructs a PyTorchVideo Kinetics
    dataset for both the train and val partitions. It defines each partition's
    augmentation and preprocessing transforms and configures the PyTorch
    DataLoaders.
    """

    def __init__(self, args):
        self.args = args
        super().__init__()
        
    def train_dataloader(self):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer 
        trains/tests with.
        """
        train_transform = MultiModeTrainDataTransform(self.args, mode="train")
        self.train_dataset = datasets.KineticsMultiMod(
            root="../content/",
            frames_per_clip=300,
            transform=train_transform,
            split="train")

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )

    def val_dataloader(self):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer 
        trains/tests with.
        """
        val_transform = MultiModeTrainDataTransform(self.args, mode="val")
        self.val_dataset = datasets.KineticsMultiMod(
            root="../content/",
            frames_per_clip=300,
            transform=val_transform,
            split="val")

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )


class LimitDataset(torch.utils.data.Dataset):
    """
    To ensure a constant number of samples are retrieved from the database
    we use this LimitDataset wrapper. This is necessary because several
    of the underlying videos may be corrupted while fetching or decoding,
    however, we always want the same number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos
