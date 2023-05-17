import argparse
import torch
import pytorch_lightning

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.plugins.environments import MPIEnvironment
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from datamodule import KineticsDataModule
from datamodule import UCF101DataModule
from datamodule import HMDB51DataModule

from model import MultiModVICRegModule
from model import EvaluatorModule

from strategy import CacheDDPStrategy

from azureml.core.run import Run


def train(args):
    """
    Network pretrain on Kinatics 400.
    """

    # Seed everything
    pytorch_lightning.seed_everything(42)

    # Callbacks
    callbacks = [LearningRateMonitor(), ModelCheckpoint(dirpath="./logs", save_last=True)]

    # MLFlow Logger
    run = Run.get_context()
    mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri() 
    mlf_logger = MLFlowLogger(experiment_name=run.experiment.name, tracking_uri=mlflow_url)
    mlf_logger._run_id = run.id

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True),
        plugins=MPIEnvironment(),
        precision=args.precision,
        gradient_clip_val=0.5,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        logger=mlf_logger,
        sync_batchnorm=True,
        use_distributed_sampler=False,
    )

    dm = KineticsDataModule(args)
    model = MultiModVICRegModule(args)

    trainer.fit(model, datamodule=dm)

    # Saving model in outputs folder
    torch.save(model, './outputs/model.pt')
    

def eval(args):
    """
    Finetune and Linear evaluation.
    """
    # Seed everything
    pytorch_lightning.seed_everything(42)

    # DataModule: Selection dataset for evaluation
    if args.eval_dataset == 'kinetics400':
        args.num_classes = 400
        dm = KineticsDataModule(args)
        
    elif args.eval_dataset == 'ucf101':
        args.num_classes = 101
        dm = UCF101DataModule(args)

    elif args.eval_dataset == 'hmdb51':
        args.num_classes = 51
        dm = HMDB51DataModule(args)
    
    # Model definition   
    model = EvaluatorModule(args)

    # Callbacks
    lr_monitor = LearningRateMonitor()

    model_checkpoint = ModelCheckpoint(
        dirpath="./logs",
        monitor="val/acc",
        save_last=True,
        mode="max",
        save_top_k = 1
    )

    callbacks = [lr_monitor, model_checkpoint] 
    
    # MLFlow Logger
    run = Run.get_context()
    mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri() 
    mlf_logger = MLFlowLogger(experiment_name=run.experiment.name, tracking_uri=mlflow_url)
    mlf_logger._run_id = run.id

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy='ddp',
        plugins=MPIEnvironment(),
        precision=args.precision,
        callbacks=callbacks,
        num_sanity_val_steps=4,
        logger=mlf_logger,
        sync_batchnorm=True,
        use_distributed_sampler=False
    )

    model = torch.compile(model, mode="reduce-overhead") 
    trainer.fit(model, datamodule=dm)
    
    # Saving model in outputs folder
    torch.save(model, './outputs/model.pt')


def main():
    """
    To train the Multi-Mode VICReg with the Kinetics dataset. 
    """
    parser = argparse.ArgumentParser()

    # Data Loader.
    parser.add_argument("--data_path", default="/home/azureuser/cloudfiles/code/fastrun", type=str)
    parser.add_argument("--video_path_prefix", default="", type=str)

    # Data Transforms
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--clip_duration", default=2, type=float)
    parser.add_argument("--temporal_distance", default=4, type=int)
    parser.add_argument("--video_num_subsampled", default=8, type=int)
    parser.add_argument("--video_means", default=(0.45, 0.45, 0.45), type=tuple)
    parser.add_argument("--video_stds", default=(0.225, 0.225, 0.225), type=tuple)
    parser.add_argument("--video_crop_size", default=224, type=int)
    parser.add_argument("--video_crop_scale", default=(0.2, 0.76), type=tuple)
    parser.add_argument("--video_min_short_side_scale", default=256, type=int)
    parser.add_argument("--video_max_short_side_scale", default=320, type=int)
    parser.add_argument("--video_horizontal_flip_p", default=0.5, type=float)
    parser.add_argument("--video_grayscale_p", default=0.2, type=float)
    parser.add_argument("--video_color_strength", default=0.75, type=float)
    parser.add_argument("--audio_raw_sample_rate", default=44100, type=int)
    parser.add_argument("--audio_resampled_rate", default=16000, type=int)
    parser.add_argument("--audio_mel_n_fft", default=1024, type=int)
    parser.add_argument("--audio_mel_win_length", default=1024, type=int)
    parser.add_argument("--audio_mel_hop_length", default=160, type=int)
    parser.add_argument("--audio_mel_n_mels", default=64, type=int)
    parser.add_argument("--audio_mel_f_min", default=60, type=int)
    parser.add_argument("--audio_mel_f_max", default=7800, type=int)
    parser.add_argument("--audio_mel_power", default=2, type=float)
    parser.add_argument("--audio_mel_time", default=96, type=float)
    parser.add_argument("--audio_logmel_mean", default=-7.03, type=float)
    parser.add_argument("--audio_logmel_std", default=4.66, type=float)

    # Backbone
    parser.add_argument("--backbone_video", default="byol_video", type=str)
    parser.add_argument("--backbone_audio", default="byol_audio", type=str)
    parser.add_argument("--path_pretrained_backbone_weights", default="/home/azureuser/cloudfiles/code/weights/byol", type=str)

    # Representations and Projections
    parser.add_argument("--intra_video_projector", default="8192-8192-8192", type=str)
    parser.add_argument("--intra_audio_projector", default="8192-8192", type=str)
    parser.add_argument("--cross_video_to_audio_projector", default="1024-512-128", type=str)
    parser.add_argument("--cross_audio_to_video_projector", default="1024-512-128", type=str)

    # Optim params
    parser.add_argument("--learning_rate", default=1.8, type=float)
    parser.add_argument("--max_epochs", default=25, type=int)
    parser.add_argument("--warmup_epochs", default=0, type=int)
    parser.add_argument("--optimizer", default="lars", type=str)
    parser.add_argument("--exclude_bn_bias", action='store_false')
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--precision", default="16-mixed", type=str)#"16-mixed"
    parser.add_argument("--num_train_samples", default=2.4e5, type=int)
    parser.add_argument("--init_backbone_freeze_epochs", default=5, type=int)

    # Loss
    parser.add_argument("--invariance-coeff", default=25.0, type=float)
    parser.add_argument("--variance-coeff", default=25.0, type=float)
    parser.add_argument("--covariance-coeff", default=1.0, type=float)
    parser.add_argument("--intra_coeff", default=1, type=float)
    parser.add_argument("--cross_coeff", default=0.2, type=float)

    # Trainer & Infrastructure
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--devices", default=8, type=int)
    parser.add_argument("--num_workers", default=5, type=int)
    parser.add_argument("--num_nodes", default=1, type=int)

    # Evaluation args
    parser.add_argument("--stage", default="pretrain", choices=["pretrain", "eval"], type=str)

    # Checkpoint & Model
    parser.add_argument("--checkpoint_path", default='./checkpoint.ckpt', type=str)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--pretrain_model_path", default='./mode.pt', type=str)

    # Eval Optim Params
    parser.add_argument("--eval_protocol", default="linear", choices=["finetune", "linear"], type=str)
    parser.add_argument("--eval_data_modality", default="video", choices=["audio", "video"], type=str)
    parser.add_argument("--eval_dataset", default="kinetics400", choices=["ucf101", "hmdb51", "kinetics400"], type=str)
    parser.add_argument("--eval_learning_rate", default=4.0, type=float)
    parser.add_argument("--eval_weight_decay", default=0.0, type=float)
    parser.add_argument("--eval_decay_epochs", default=(60,120,180), type=tuple)
    parser.add_argument("--eval_gamma", default=0.1, type=float)
    parser.add_argument("--eval_dropout_p", default=0.5, type=float)
    parser.add_argument("--eval_scheduler_type", default='cosine', type=str)

    args = parser.parse_args()
    
    if args.stage == 'pretrain':
        train(args)
    elif args.stage == 'eval':
        eval(args)


if __name__ == "__main__":
    main()
