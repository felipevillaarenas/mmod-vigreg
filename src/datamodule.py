import itertools
import argparse

import os

import pytorch_lightning
import pytorchvideo.data
import torch

from torch.utils.data import DistributedSampler, RandomSampler

from transforms import MultiModeTrainDataTransform

class KineticsDataModule(pytorch_lightning.LightningDataModule):
    """
    This LightningDataModule implementation constructs a PyTorchVideo Kinetics dataset for both
    the train and val partitions. It defines each partition's augmentation and
    preprocessing transforms and configures the PyTorch DataLoaders.
    """

    def __init__(self, args):
        self.args = args
        super().__init__()

    def prepare_data(self):
        # download
        #torchvision.datasets.Kinetics(self.data_dir, split="val", num_classes=self.num_classes,download=True)
        pass

    def train_dataloader(self):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer trains/tests with.
        """
        sampler = RandomSampler#DistributedSampler if self.trainer.use_ddp else RandomSampler
        train_transform = MultiModeTrainDataTransform(self.args,mode="train")
        self.train_dataset = LimitDataset(
            pytorchvideo.data.Kinetics(
                data_path=os.path.join(self.args.data_path, "train"),
                clip_sampler=pytorchvideo.data.make_clip_sampler(
                    "random", self.args.total_clip_duration
                ),
                video_path_prefix=self.args.video_path_prefix,
                transform=train_transform,
                video_sampler=sampler,
            )
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )

    def val_dataloader(self):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer trains/tests with.
        """
        sampler = RandomSampler#DistributedSampler if self.trainer.use_ddp else RandomSampler
        val_transform = MultiModeTrainDataTransform(self.args,mode="val")
        self.val_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self.args.data_path, "val"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform", self.args.total_clip_duration
            ),
            video_path_prefix=self.args.video_path_prefix,
            transform=val_transform,
            video_sampler=sampler,
        )
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data parameters.
    parser.add_argument("--data_path", default="./data/kinetics400small/", type=str)
    parser.add_argument("--video_path_prefix", default="", type=str)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--subsample_clip_duration", default=1, type=float)
    parser.add_argument("--total_clip_duration", default=10, type=float)
    parser.add_argument("--data_type", default="video-audio", choices=["video", "audio"], type=str)
    parser.add_argument("--video_num_subsampled", default=16, type=int)
    parser.add_argument("--video_means", default=(0.45, 0.45, 0.45), type=tuple)
    parser.add_argument("--video_stds", default=(0.225, 0.225, 0.225), type=tuple)
    parser.add_argument("--video_crop_size", default=224, type=int)
    parser.add_argument("--video_min_short_side_scale", default=256, type=int)
    parser.add_argument("--video_max_short_side_scale", default=320, type=int)
    parser.add_argument("--video_horizontal_flip_p", default=0.5, type=float)
    parser.add_argument("--audio_raw_sample_rate", default=44100, type=int)
    parser.add_argument("--audio_resampled_rate", default=16000, type=int)
    parser.add_argument("--audio_mel_window_size", default=32, type=int)
    parser.add_argument("--audio_mel_step_size", default=16, type=int)
    parser.add_argument("--audio_num_mels", default=80, type=int)
    parser.add_argument("--audio_mel_num_subsample", default=128, type=int)
    parser.add_argument("--audio_logmel_mean", default=-7.03, type=float)
    parser.add_argument("--audio_logmel_std", default=4.66, type=float)

    args = parser.parse_args()

    dm = KineticsDataModule(args)
    dm.val_dataloader()
    sample = dm.val_dataset.__next__()
    audio = sample['audio']
    video = sample['video']

    print("run exp")