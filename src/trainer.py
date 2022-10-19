import argparse

import pytorch_lightning

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from datamodule_map.datamodule import KineticsDataModule
from model import MultiModVICRegModule


def main():
    """
    To train the ResNet with the Kinetics dataset we construct the two modules
    above, and pass them to the fit function of a pytorch_lightning.Trainer.
    This example can be run either locally (with default parameters) or on
    a Slurm cluster. To run on a Slurm cluster provide the --on_cluster
    argument.
    """

    pytorch_lightning.trainer.seed_everything()
    parser = argparse.ArgumentParser()

    # Data Loader.
    parser.add_argument("--data_path", default="./data/kinetics400small/", type=str)
    parser.add_argument("--video_path_prefix", default="", type=str)

    # Data Transforms
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--subsample_clip_duration", default=1, type=float)
    parser.add_argument("--total_clip_duration", default=10, type=float)
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

    # Representations and Projections
    parser.add_argument("--video_representations_dim", default=2048, type=int)
    parser.add_argument("--audio_representations_dim", default=2048, type=int)
    parser.add_argument("--intra_video_projector", default="4096-4096", type=str)
    parser.add_argument("--intra_audio_projector", default="4096-4096", type=str)
    parser.add_argument("--cross_video_to_audio_projector", default="1024-256", type=str)
    parser.add_argument("--cross_audio_to_video_projector", default="1024-256", type=str)

    # Optim params
    parser.add_argument("--optimizer", default="lars", type=str)
    parser.add_argument("--exclude_bn_bias", default=False, type=bool)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--learning_rate", default=0.3, type=float)
    parser.add_argument("--max_epochs", default=1, type=int)
    parser.add_argument("--fp32", default=False, type=bool)
    parser.add_argument("--warmup_epochs", default=10, type=int)

    # Loss
    parser.add_argument("--invariance-coeff", default=25.0, type=float)
    parser.add_argument("--variance-coeff", default=25.0, type=float)
    parser.add_argument("--covariance-coeff", default=1.0, type=float)
    parser.add_argument("--intra_coeff", default=10.0, type=float)
    parser.add_argument("--cross_coeff", default=1.0, type=float)

    # Trainer & Infrastructure
    parser.add_argument("--accelerator", default="auto", type=str)
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--nodes", default=1, type=int)

    # Online eval
    parser.add_argument("--online_ft", default=True, type=bool)

    # Build trainer, ResNet lightning-module and Kinetics data-module.
    args = parser.parse_args()

    train(args)


def train(args):
    model = MultiModVICRegModule(args)
    dm = KineticsDataModule(args)

    # Distributed params
    args.num_samples = 100
    # args.num_samples = dm.train_dataloader().dataset.dataset.num_videos
    if args.devices > 0:
        args.global_batch_size = args.nodes * args.devices * args.batch_size
        args.train_iters_per_epoch = args.num_samples // args.global_batch_size
    else:
        args.batch_size
        args.train_iters_per_epoch = args.num_samples // args.global_batch_size

    # Scheduler params
    args.warmup_steps = args.warmup_epochs * args.train_iters_per_epoch
    args.total_steps = args.max_epochs * args.train_iters_per_epoch

    callbacks = list()

    # Callback learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Callback model checkpoint
    model_checkpoint = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        monitor="train/loss"
    )
    callbacks.append(model_checkpoint)

    trainer = Trainer(
        profiler="simple",
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        strategy="ddp" if args.devices > 1 else None,
        sync_batchnorm=True if args.devices > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    
    main()

