import argparse
import logging


import pytorch_lightning

from pytorch_lightning.callbacks import LearningRateMonitor

from model import VideoClassificationLightningModule
from datamodule import KineticsDataModule

def main():
    """
    To train the ResNet with the Kinetics dataset we construct the two modules above,
    and pass them to the fit function of a pytorch_lightning.Trainer.
    This example can be run either locally (with default parameters) or on a Slurm
    cluster. To run on a Slurm cluster provide the --on_cluster argument.
    """
    setup_logger()

    pytorch_lightning.trainer.seed_everything()
    parser = argparse.ArgumentParser()

    #  Cluster parameters.
    parser.add_argument("--on_cluster", action="store_true")
    parser.add_argument("--job_name", default="ptv_video_classification", type=str)
    parser.add_argument("--working_directory", default=".", type=str)
    parser.add_argument("--partition", default="dev", type=str)

    # Model parameters.
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument(
        "--arch",
        default="video_resnet",
        choices=["video_resnet", "audio_resnet"],
        type=str,
    )

    # Data parameters.
    parser.add_argument("--data_path", default="./data/video/kinetics400/", type=str)
    parser.add_argument("--video_path_prefix", default="", type=str)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--clip_duration", default=2, type=float)
    parser.add_argument(
        "--data_type", default="video", choices=["video", "audio"], type=str
    )
    parser.add_argument("--video_num_subsampled", default=8, type=int)
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

    # Trainer parameters.
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=200,
        callbacks=[LearningRateMonitor()],
        replace_sampler_ddp=False,
    )

    # Build trainer, ResNet lightning-module and Kinetics data-module.
    args = parser.parse_args()
    train(args)


def train(args):
    trainer = pytorch_lightning.Trainer.from_argparse_args(args)
    classification_module = VideoClassificationLightningModule(args)
    data_module = KineticsDataModule(args)
    trainer.fit(classification_module, data_module)


def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("pytorchvideo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)


if __name__ == "__main__":
    main()