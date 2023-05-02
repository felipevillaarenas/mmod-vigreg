import os
import itertools

import pytorch_lightning
import pytorchvideo.data
import torch

from torch.utils.data import RandomSampler, DistributedSampler

from transforms import MultiModeTrainDataTransform
from transforms import EvalDataTransform


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
        Defines the train DataLoader for the Kinetics dataset.
        """
        sampler = DistributedSampler if self.args.num_nodes * self.args.devices > 1 else RandomSampler 

        train_transform = MultiModeTrainDataTransform(self.args, mode="train")
        clip_duration_pretrain = self.args.temporal_distance * self.args.clip_duration

        self.train_dataset =  LimitDataset(
            pytorchvideo.data.Kinetics(
                data_path=os.path.join(self.args.data_path, "train"),
                clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration_pretrain),
                video_path_prefix=self.args.video_path_prefix,
                transform=train_transform,
                video_sampler=sampler,
            )
        )

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory= True,
            drop_last=True
            )



class UCF101DataModule(pytorch_lightning.LightningDataModule):
    """
    This LightningDataModule implementation constructs a PyTorchVideo UCF-101
    dataset for both the train and val partitions. It defines each partition's
    augmentation and preprocessing transforms and configures the PyTorch
    DataLoaders.
    """

    def __init__(self, args):
        self.args = args
        super().__init__()

    def train_dataloader(self):
        """
        Defines the train DataLoader for the UCF-101 dataset.
        """
        sampler = DistributedSampler if self.args.num_nodes * self.args.devices > 1 else RandomSampler 

        train_transform = EvalDataTransform(self.args, mode="train")
        
        self.train_dataset =  LimitDataset(
            pytorchvideo.data.Ucf101(
                data_path=os.path.join(self.args.data_path, "annotations/trainlist01.txt"),
                clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.args.clip_duration),
                video_path_prefix=os.path.join(self.args.data_path, "videos/"),
                transform=train_transform,
                video_sampler=sampler,
                decode_audio=False
            )
        )

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory= True,
            drop_last=True
            )

    def val_dataloader(self):
        """
        Defines the val DataLoader for the UCF-101 dataset.
        """
        sampler = DistributedSampler if self.args.num_nodes * self.args.devices > 1 else RandomSampler 

        val_transform = EvalDataTransform(self.args, mode="val")
        
        self.val_dataset =  LimitDataset(
            pytorchvideo.data.Ucf101(
                data_path=os.path.join(self.args.data_path, "annotations/testlist01.txt"),
                clip_sampler=pytorchvideo.data.make_clip_sampler("constant_clips_per_video", self.args.clip_duration, 1, 1),
                video_path_prefix=os.path.join(self.args.data_path, "videos/"),
                transform=val_transform,
                video_sampler=sampler,
                decode_audio=False
            )
        )
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory= True,
            drop_last=True
            )

    def test_dataloader(self):
        """
        Defines the test DataLoader for the UCF-101 dataset.
        """
        sampler = DistributedSampler if self.args.num_nodes * self.args.devices > 1 else RandomSampler 

        test_transform = EvalDataTransform(self.args, mode="val")
        
        self.test_dataset =  LimitDataset(
            pytorchvideo.data.Ucf101(
                data_path=os.path.join(self.args.data_path, "annotations/testlist01.txt"),
                clip_sampler=pytorchvideo.data.make_clip_sampler("constant_clips_per_video", self.args.clip_duration, 10, 3),
                video_path_prefix=os.path.join(self.args.data_path, "videos/"),
                transform=test_transform,
                video_sampler=sampler,
                decode_audio=False
            )
        )
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory= True,
            drop_last=True
            )

class HMDB51DataModule(pytorch_lightning.LightningDataModule):
    """
    This LightningDataModule implementation constructs a PyTorchVideo HMDB51
    dataset for both the train and val partitions. It defines each partition's
    augmentation and preprocessing transforms and configures the PyTorch
    DataLoaders.
    """

    def __init__(self, args):
        self.args = args
        data_path=os.path.join(self.args.data_path, "annotations")
        super().__init__()

    def train_dataloader(self):
        """
        Defines the train DataLoader for the HMDB51 dataset.
        """
        sampler = DistributedSampler if self.args.num_nodes * self.args.devices > 1 else RandomSampler 

        train_transform = EvalDataTransform(self.args, mode="train")
        
        self.train_dataset = LimitDataset(
            pytorchvideo.data.Hmdb51(
                data_path=os.path.join(self.args.data_path, "annotations"),
                clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.args.clip_duration),
                split_id=1,
                split_type="train",
                video_path_prefix=os.path.join(self.args.data_path, "videos"),
                transform=train_transform,
                video_sampler=sampler,
                decode_audio=False
            )
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory= True,
            drop_last=True,
            )

    def val_dataloader(self):
        """
        Defines the val DataLoader for the HMDB51 dataset.
        """
        sampler = DistributedSampler if self.args.num_nodes * self.args.devices > 1 else RandomSampler 

        val_transform = EvalDataTransform(self.args, mode="val")
        
        self.val_dataset =  LimitDataset(
            pytorchvideo.data.Hmdb51(
                data_path=os.path.join(self.args.data_path, "annotations"),
                clip_sampler=pytorchvideo.data.make_clip_sampler("constant_clips_per_video",self.args.clip_duration, 1, 1),
                split_id=1,
                split_type="test",
                video_path_prefix=os.path.join(self.args.data_path, "videos"),
                transform=val_transform,
                video_sampler=sampler,
                decode_audio=False
            )
        )
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory= True,
            drop_last=True
            )
    
    def test_dataloader(self):
        """
        Defines the test DataLoader for the HMDB51 dataset.
        """
        sampler = DistributedSampler if self.args.num_nodes * self.args.devices > 1 else RandomSampler 

        test_transform = EvalDataTransform(self.args, mode="val")
        
        self.test_dataset =  LimitDataset(
            pytorchvideo.data.Hmdb51(
                data_path=os.path.join(self.args.data_path, "annotations"),
                clip_sampler=pytorchvideo.data.make_clip_sampler("constant_clips_per_video", self.args.clip_duration, 10, 3),
                split_id=1,
                split_type="test",
                video_path_prefix=os.path.join(self.args.data_path, "videos"),
                transform=test_transform,
                video_sampler=sampler,
                decode_audio=False
            )
        )
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory= True,
            drop_last=True
            )


class LimitDataset(torch.utils.data.Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
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