import random

import torch

from pytorchvideo.transforms import (
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
    Div255
)
from torchaudio.transforms import MelSpectrogram, Resample
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)


class MultiModeTrainDataTransform:
    """ Video Transforms for Multi Modal Self Supervised Learning."""

    def __init__(self, mode):

        super().__init__()
        self.total_clip_duration = 2
        self.subsample_clip_duration = 1
        self.video = VideoTrainDataTransform(mode)
        self.audio = AudioTrainDataTransform(mode)

    def __call__(self, sample):
        sample

        # Chunk number
        chunks = int(
            self.total_clip_duration / self.subsample_clip_duration
        )

        # Temporal Clips
        videos = torch.chunk(sample['video'], chunks=chunks, dim=1)
        audios = torch.chunk(sample['audio'], chunks=chunks, dim=0)

        # Random clip idxs
        chunk_ids = [i for i in range(chunks)]
        idx = random.choice(chunk_ids)
        chunk_ids.remove(idx)
        idx_prime = random.choice(chunk_ids)

        # Applying transforms
        sample['video'] = (
            self.video.transform(videos[idx]),
            self.video.transform(videos[idx_prime])
        )
        sample['audio'] = (
            self.audio.transform(audios[idx]),
            self.audio.transform(audios[idx_prime])
            )

        return sample


class VideoTrainDataTransform:
    """ Video Transforms for Multi Modal Self Supervised Learning."""

    def __init__(self, mode):

        super().__init__()
        self.video_num_subsampled = 16
        self.video_means = (0.45, 0.45, 0.45)
        self.video_stds = (0.225, 0.225, 0.225)
        self.video_min_short_side_scale = 256
        self.video_max_short_side_scale = 320
        self.video_crop_size = 224
        self.video_horizontal_flip_p = 0.5

        self.transform = Compose(
            [
                UniformTemporalSubsample(
                    num_samples=self.video_num_subsampled,
                    temporal_dim=1
                ),
                Div255(),
                Normalize(
                    mean=self.video_means,
                    std=self.video_stds
                ),
            ]
            + (
                [
                    RandomShortSideScale(
                        min_size=self.video_min_short_side_scale,
                        max_size=self.video_max_short_side_scale,
                    ),
                    RandomCrop(size=self.video_crop_size),
                    RandomHorizontalFlip(p=self.video_horizontal_flip_p),
                ]
                if mode == "train"
                else [
                    ShortSideScale(size=self.video_min_short_side_scale),
                    CenterCrop(size=self.video_crop_size),
                ]
            )
        )


class AudioTrainDataTransform:
    """ Audio Transforms for Multi Modal Self Supervised Learning."""

    def __init__(self, mode):

        super().__init__()
        self.audio_raw_sample_rate = 44100
        self.audio_resampled_rate = 16000
        self.audio_mel_window_size = 32
        self.audio_mel_step_size = 16
        self.audio_num_mels = 80
        self.audio_mel_num_subsample = 128
        self.audio_logmel_mean = -7.03
        self.audio_logmel_std = 4.66

        self.transform = Compose(
            [
                Resample(
                    orig_freq=self.audio_raw_sample_rate,
                    new_freq=self.audio_resampled_rate,
                ),
                MelSpectrogram(
                    sample_rate=self.audio_resampled_rate,
                    n_fft=int(
                        float(self.audio_resampled_rate) / 1e3
                        * self.audio_mel_window_size
                        ),
                    hop_length=int(
                        float(self.audio_resampled_rate) / 1e3
                        * self.audio_mel_step_size),
                    n_mels=self.audio_num_mels,
                    center=False,
                ),
                Lambda(Clamp(min=1e-10)),  #lambda x: x.clamp(min=1e-10)),
                Lambda(torch.log),
                UniformTemporalSubsample(
                    num_samples=self.audio_mel_num_subsample,
                    temporal_dim=1
                ),
                Lambda(AudioReshape()),  # lambda(lambda x: x.transpose(1, 0)),lambda x: x.view(1, x.size(0), 1, x.size(1))),
                Normalize((self.audio_logmel_mean,), (self.audio_logmel_std,)),
            ]
        )


class Clamp:
    def __init__(self, min: float):
        self.min = min

    def __call__(self, x):
        return x.clamp(min=self.min)


class AudioReshape:
    def __init__(self, val=None):
        self.val = None

    def __call__(self, x):
        x = x.transpose(1, 0)
        return x.view(1, x.size(0), 1, x.size(1))
