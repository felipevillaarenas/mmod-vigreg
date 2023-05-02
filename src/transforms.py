import torch

from pytorchvideo.transforms import (
    Normalize,
    ShortSideScale,
    UniformTemporalSubsample,
    RandomShortSideScale,
    Div255
)
from torchaudio.transforms import MelSpectrogram, Resample
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomHorizontalFlip,
    RandomCrop,
    RandomApply,
    ColorJitter,
    RandomGrayscale,
)


class MultiModeTrainDataTransform:
    """ Video Transforms for Multi Modal Self Supervised Learning."""

    def __init__(self, args, mode):

        super().__init__()
        self.args = args
        self.video = VideoTrainDataTransform(args, mode)
        self.audio = AudioTrainDataTransform(args, mode)

    def __call__(self, sample):

        # Clip idxs
        idx, idx_prime = (0, self.args.temporal_distance - 1)
        

        # Temporal Clips
        try:
            videos = torch.chunk(sample['video'], chunks=self.args.temporal_distance , dim=1)
            
            sample['video'] = (
                self.video.transform(videos[idx]),
                self.video.transform(videos[idx_prime])
            )

        except:
            print('empty video')
            # Shape: (C, T, H, W)
            channels = 3
            video_empty = 255 * torch.rand(
                [
                    channels,
                    int(self.args.clip_duration * self.args.video_num_subsampled),
                    self.args.video_min_short_side_scale,
                    self.args.video_min_short_side_scale
                ]
            )
            videos = tuple()
            for i in range(self.args.temporal_distance):
                videos += (video_empty,)
            sample['video'] = (
                self.video.transform(videos[idx]),
                self.video.transform(videos[idx_prime])
            )

        try:
            audios = torch.chunk(sample['audio'], chunks=self.args.temporal_distance , dim=0)
            sample['audio'] = (
                self.audio.transform(audios[idx]),
                self.audio.transform(audios[idx_prime])
                )
        except:
            print('empty audio')
            audio_empty = 0.1 * (torch.rand([self.args.audio_raw_sample_rate]) - 0.5)
            audios = tuple()
            for i in range(self.args.temporal_distance):
                audios += (audio_empty,)
            sample['audio'] = (
                self.audio.transform(audios[idx]),
                self.audio.transform(audios[idx_prime])
                )     
        return sample


class EvalDataTransform:
    """ Video Transforms for Fine Tuning and Linear Evaluation."""

    def __init__(self, args, mode):

        super().__init__()
        self.args = args
        self.video = VideoTrainDataTransform(args, mode)
        self.audio = AudioTrainDataTransform(args, mode)

    def __call__(self, sample):
        # Applying transforms
        if self.args.eval_data_modality == 'video':
            sample['video'] = self.video.transform(sample['video'])

        elif self.args.eval_data_modality == 'audio':
            sample['audio'] = self.audio.transform(sample['audio'])

        return sample


class VideoTrainDataTransform:
    """ Video Transforms for Multi Modal Self Supervised Learning."""

    def __init__(self, args, mode):

        super().__init__()

        self.transform = Compose(
            [
                UniformTemporalSubsample(
                    num_samples= int(args.clip_duration * args.video_num_subsampled),
                    temporal_dim=1
                ),
                Div255(),
            ]
            + (
                [ 
                    RandomShortSideScale(
                        min_size=args.video_min_short_side_scale,
                        max_size=args.video_max_short_side_scale,
                    ),
                    RandomCrop(size=args.video_crop_size),
                    RandomHorizontalFlip(p=args.video_horizontal_flip_p),
                    PermuteChannelPerFrames(),
                    RandomApply(
                        [
                            ColorJitter(
                                brightness=0.8 * args.video_color_strength, 
                                contrast=0.8 * args.video_color_strength,
                                saturation=0.8 * args.video_color_strength,
                                hue=0.2 * args.video_color_strength
                            )
                        ],
                        p=0.8,
                    ),
                    RandomGrayscale(p=args.video_grayscale_p),
                    PermuteChannelPerFrames(),
                    Normalize(
                        mean=args.video_means,
                        std=args.video_stds
                    )
                ]
                if mode == "train"
                else [
                    ShortSideScale(size=args.video_min_short_side_scale),
                    CenterCrop(size=args.video_crop_size),
                    Normalize(
                        mean=args.video_means,
                        std=args.video_stds
                    )
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
                AudioCut(samples = args.audio_resampled_rate),
                MelSpectrogram(
                    sample_rate=args.audio_resampled_rate,
                    n_fft=int(
                        float(args.audio_resampled_rate) / 1e3
                        * args.audio_mel_window_size
                        ),
                    hop_length=int(
                        float(args.audio_resampled_rate) / 1e4
                        * args.audio_mel_step_size),
                    n_mels=args.audio_num_mels,
                    center=False,
                    power=args.audio_mel_power
                ),
                Lambda(Clamp(min=1e-10)),
                Lambda(torch.log),
                Lambda(AudioReshape()),
                Normalize((args.audio_logmel_mean,), (args.audio_logmel_std,)),
                Lambda(Squeeze(dim=0)),
            ]
        )


class Clamp:
    """Audio Clap transform"""
    def __init__(self, min: float):
        self.min = min

    def __call__(self, x):
        return x.clamp(min=self.min)
    

class AudioCut:
    """Audio reshape transform"""
    def __init__(self, samples):
        self.samples = samples

    def __call__(self, x):
        return x[:self.samples]


class AudioReshape:
    """Audio reshape transform"""
    def __init__(self):
        self.min = 1e-10

    def __call__(self, x):
        return x.view(1, 1, x.size(0), x.size(1))


class Squeeze:
    """Audio squeeze transform"""
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, x):
        return x.squeeze(self.dim)


class PermuteChannelPerFrames(torch.nn.Module):
    """
    A Scriptable version to perform:
    - (C, T, H, W) > (T, C, H, W)
    - (T, C, H, W) > (C, T, H, W) 
    """

    def __init__(self):
        pass

    def __call__(self, x):
        x = x.permute([1, 0, 2, 3])
        return x