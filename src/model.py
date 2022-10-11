import pytorch_lightning
import torch
import torch.nn as nn

from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pytorchvideo.models.resnet import create_resnet, create_acoustic_resnet

from loss import VICRegLoss


class MultiModVICRegModule(pytorch_lightning.LightningModule):
    """
    PyTorch Lightning implementation of Multi-Modal VICReg.
    """
    def __init__(self, args,):
        
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        # Init backbone
        self.video_backbone = self.init_video_backbone()
        self.audio_backbone = self.init_audio_backbone()

        # Init Projector
        self.intra_video_projector = self.init_projector(
            args.video_representations_dim,
            args.intra_video_projector
        )

        self.intra_audio_projector = self.init_projector(
            args.audio_representations_dim,
            args.intra_audio_projector
        )

        self.cross_audio_to_video_projector = self.init_projector(
            args.audio_representations_dim,
            args.cross_audio_to_video_projector
        )

        self.cross_video_to_audio_projector = self.init_projector(
            args.video_representations_dim,
            args.cross_video_to_audio_projector
        )

        # Init loss
        self.loss = VICRegLoss()
        
    def init_video_backbone(self):
        """
        Create video backbone.
        """
        video_backbone = create_resnet(
                input_channel=3,
                model_num_class=2048,
            )
        return video_backbone

    def init_audio_backbone(self):
        """
        Create audio backbone.
        """
        audio_backbone = create_acoustic_resnet(
                input_channel=1,
                model_num_class=2048
            )
        return audio_backbone
    
    def init_projector(self, representations_dim, projector_dims):
        """
        Creates projection head.

        Args:
            representations_dim (int): latent dimension.
            projector_dims (str): Dimension projection head.
        """
        mlp_spec = f"{representations_dim}-{projector_dims}"
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward defines the prediction/inference actions.
        """
        return (self.video_backbone(x['video']), self.audio_backbone(x['audio']))

    def share_step(self, batch, batch_idx):
        """
        Calculates total, intra modality and cross modality losses.

        Returns:
            dict: Summary of losses.
        """
        intra_video_loss, video_reps = self.intra_video_step(batch['video'])
        intra_audio_loss, audio_reps = self.intra_audio_step(batch['audio'])

        video_rep1, video_rep2 = video_reps
        audio_rep1, audio_rep2 = audio_reps

        cross_video_audio_loss = (
            self.cross_video_audio_step(video_rep1, audio_rep1) +
            self.cross_video_audio_step(video_rep2, audio_rep2)
        )

        loss = (
            self.args.intra_coeff * intra_video_loss
            + self.args.intra_coeff * intra_audio_loss
            + self.args.cross_coeff * cross_video_audio_loss
        )
        return {
            'loss': loss,
            'intra_video_loss': intra_video_loss,
            'intra_audio_loss': intra_audio_loss,
            'cross_video_audio_loss': cross_video_audio_loss
        }

    def intra_video_step(self, samples):
        """
        Calculates the video representations of clips from the same video.
        Such representations are expanded using a projection head. Finally,
        the variance, invariance and covariance loss is calculated.

        Args:
            samples (tuple): Random video clips from the same video. 

        Returns:
            tuple containing:
                - intra_video_loss: VICReg Loss for video intra modality.
                - (video_rep1, video_rep2): Video representations for clip1 and clip2.
        """
        # video: batches of transform views
        video1, video2 = samples

        # video: batches of representations
        video_rep1 = self.video_backbone(video1)
        video_rep2 = self.video_backbone(video2)

        # video: batches of embeddings
        video_emb1 = self.intra_video_projector(video_rep1)
        video_emb2 = self.intra_video_projector(video_rep2)

        # loss intra video modality
        intra_video_loss = self.loss(video_emb1, video_emb2)
        return intra_video_loss, (video_rep1, video_rep2)

    def intra_audio_step(self, samples):
        """
        Calculates the audio representations of clips from the same audio.
        Such representations are expanded using a projection head. Finally,
        the variance, invariance and covariance loss is calculated.

        Args:
            samples (tuple): Random audio clips from the same audio.

        Returns:
            tuple containing:
                - intra_audio_loss: VICReg Loss for audio intra modality.
                - (audio_rep1, audio_rep2): Audio representations for clip1 and clip2.
        """
        # audio: batches of transform views
        audio1, audio2 = samples

        # audio: batches of representations
        audio_rep1 = self.audio_backbone(audio1)
        audio_rep2 = self.audio_backbone(audio2)

        # audio: batches of embeddings
        audio_emb1 = self.intra_audio_projector(audio_rep1)
        audio_emb2 = self.intra_audio_projector(audio_rep2)

        # loss intra audio modality
        intra_audio_loss = self.loss(audio_emb1, audio_emb2)
        return intra_audio_loss, (audio_rep1, audio_rep2)
    
    def cross_video_audio_step(self, video_rep, audio_rep):
        """
        Project video and audio using a projection head to calculate
        intra modality VICReg loss.

        Args:
            video_rep (tensor): Video representation.
            audio_rep (tensor): Audio representation.

        Returns:
            cross_video_audio_loss: VICReg Loss for video-audio cross modality.
        """
        # loss intra audio modality
        cross_video_audio_emb = self.cross_video_to_audio_projector(video_rep)
        cross_audio_video_emb = self.cross_audio_to_video_projector(audio_rep)
        cross_video_audio_loss = self.loss(cross_video_audio_emb, cross_audio_video_emb)
        return cross_video_audio_loss

    def training_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the training epoch.
        It must return a loss that is used for loss.backwards() internally.
        PyTorchVideo batches are dictionaries containing each modality
        or metadata of the batch collated video clips. Kinetics contains
        the following notable keys:
           {
               'video': <video_tensor>,
               'audio': <audio_tensor>,
               'label': <action_label>,
           }
        - "video" is a Tensor of shape (batch, channels, time, height, Width)
        - "audio" is a Tensor of shape (batch, channels, time, 1, frequency)
        - "label" is a Tensor of shape (batch, 1)
        The PyTorchVideo models and transforms expect the same input shapes and
        dictionary structure making this function just a matter of unwrapping
        the dict and feeding it through the model/loss.
        """
        losses = self.share_step(batch, batch_idx)

        # log results
        self.log_dict(
            {
                "train/loss": losses['loss'],
                "train/intra_video_loss": losses['intra_video_loss'],
                "train/intra_audio_loss": losses['intra_audio_loss'],
                "train/cross_video_audio_loss": losses['cross_video_audio_loss'],
            }
        )
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the evaluation cycle.
        For this simple example it's mostly the same as the training loop
        but with a different metric name.
        """
        losses = self.share_step(batch, batch_idx)

        # log results
        self.log_dict(
            {
                "val/loss": losses['loss'],
                "val/intra_video_loss": losses['intra_video_loss'],
                "val/intra_audio_loss": losses['intra_audio_loss'],
                "val/cross_video_audio_loss": losses['cross_video_audio_loss'],
            }
        )
        return losses['loss']

    def configure_optimizers(self):
        """
        We use the LARS/Adam optimizer with per step cosine annealing
        scheduler.
        """
        params = self.parameters()

        # Optimizer
        if self.args.optimizer == "lars":
            optimizer = LARS(
                params,
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=self.args.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )

        # Scheduler
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(
                    self.args.warmup_steps,
                    self.args.total_steps,
                    cosine=True
                ),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

