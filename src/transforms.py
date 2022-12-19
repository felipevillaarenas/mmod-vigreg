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
        self.video = VideoTrainDataTransform(args, mode)
        self.audio = AudioTrainDataTransform(args, mode)

    def __call__(self, sample):

        # Temporal Clips
        videos = torch.chunk(sample['video'], chunks=2, dim=1)
        audios = torch.chunk(sample['audio'], chunks=2, dim=0)

        # Random clip idxs
        idx, idx_prime = (0, 1)

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

    def __init__(self, args, mode):

        super().__init__()

        self.transform = Compose(
            [
                UniformTemporalSubsample(
                    num_samples=args.video_num_subsampled,
                    temporal_dim=1
                ),
                Div255(),
                Normalize(
                    mean=args.video_means,
                    std=args.video_stds
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


class AudioTrainDataTransform:
    """ Audio Transforms for Multi Modal Self Supervised Learning."""

    def __init__(self, args, mode):

        super().__init__()
        self.args = args
        self.transform = Compose(
            [
                Resample(
                    orig_freq=args.audio_raw_sample_rate,
                    new_freq=args.audio_resampled_rate,
                ),
                MelSpectrogram(
                    sample_rate=args.audio_resampled_rate,
                    n_fft=int(
                        float(args.audio_resampled_rate) / 1e3
                        * args.audio_mel_window_size
                        ),
                    hop_length=int(
                        float(args.audio_resampled_rate) / 1e3
                        * args.audio_mel_step_size),
                    n_mels=args.audio_num_mels,
                    center=False,
                ),
                Lambda(Clamp(min=1e-10)),
                Lambda(torch.log),
                UniformTemporalSubsample(
                    num_samples=args.audio_mel_num_subsample,
                    temporal_dim=1
                ),
                Lambda(AudioReshape()),
                Normalize((args.audio_logmel_mean,), (args.audio_logmel_std,)),
                Lambda(Squeeze(dim=0)),
            ]
        )


class Clamp:
    def __init__(self, min: float):
        self.min = min

    def __call__(self, x):
        return x.clamp(min=self.min)


class AudioReshape:
    def __init__(self):
        self.min = 1e-10

    def __call__(self, x):
        return x.view(1, 1, x.size(1), x.size(0))


class Squeeze:
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, x):
        return x.squeeze(self.dim)
