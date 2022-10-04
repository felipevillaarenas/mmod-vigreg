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

    def __init__(self, args, mode):

        super().__init__()
        self.args = args

        self.video_transform = Compose(
            [
                UniformTemporalSubsample(
                    num_samples=args.video_num_subsampled,
                    temporal_dim=1
                ),
                Div255(),
                Normalize(
                    mean=args.video_means,
                    std=args.video_stdvideos
                ),
            ]
            + (
                [
                    RandomShortSideScale(
                        min_size=args.video_min_short_side_scale,
                        max_size=args.video_max_short_side_scale,
                    ),
                    RandomCrop(size=args.video_crop_size),
                    RandomHorizontalFlip(p=args.video_horizontal_flip_p),
                ]
                if mode == "train"
                else [
                    ShortSideScale(size=args.video_min_short_side_scale),
                    CenterCrop(size=args.video_crop_size),
                ]
            )
        )

        self.audio_transform = Compose(
            [
                Resample(
                    orig_freq=args.audio_raw_sample_rate,
                    new_freq=args.audio_resampled_rate,
                ),
                MelSpectrogram(
                    sample_rate=args.audio_resampled_rate,
                    n_fft=int(
                        float(args.audio_resampled_rate) / 1000
                        * args.audio_mel_window_size
                    ),
                    hop_length=int(
                        float(args.audio_resampled_rate) / 1000
                        * args.audio_mel_step_size
                    ),
                    n_mels=args.audio_num_mels,
                    center=False,
                ),
                Lambda(lambda x: x.clamp(min=1e-10)),
                Lambda(torch.log),
                UniformTemporalSubsample(
                    num_samples=args.audio_mel_num_subsample,
                    temporal_dim=1
                ),
                Lambda(lambda x: x.transpose(1, 0)),
                Lambda(lambda x: x.view(1, x.size(0), 1, x.size(1))),
                Normalize((args.audio_logmel_mean,), (args.audio_logmel_std,)),
            ]
        )

    def __call__(self, sample):
        # Chunk number
        chunks = int(
            self.args.total_clip_duration / self.args.subsample_clip_duration
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
            self.video_transform(videos[idx]),
            self.video_transform(videos[idx_prime])
        )
        sample['audio'] = (
            self.audio_transform(audios[idx]),
            self.audio_transform(audios[idx_prime])
        )
        return sample
