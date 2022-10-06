import pytorch_lightning
import torch
import torch.nn.functional as F
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

        # Init loss
        self.intra_video_loss = VICRegLoss(
            self.args.batch_size,
            self.args.num_features_intra_video
        )
        self.intra_audio_loss = VICRegLoss(
            self.args.batch_size,
            self.args.num_features_intra_video
        )
        self.cross_audio_video_loss = VICRegLoss(
            self.args.batch_size,
            self.args.num_features_cross_audio_video
        )

    def init_video_backbone(self):
        video_backbone = create_resnet(
                input_channel=3,
                model_num_class=400,
            )
        return video_backbone

    def init_audio_backbone(self):
        audio_backbone = create_acoustic_resnet(
                input_channel=1,
                model_num_class=400,
            )
        return audio_backbone

    def forward(self, x):
        """
        Forward defines the prediction/inference actions.
        """
        return (self.video_backbone(x['video']), self.audio_backbone(x['audio']))

    def intra_video_step(self, batch, batch_idx):
        # x1, x2: batches of transform views
        (x1, x2) = batch

        # y1, y2: batches of representations
        y1 = self.video_backbone(x1['video'])
        y2 = self.video_backbone(x2['video'])

        # z1, z2: batches of embeddings
        z1 = self.intra_video_projector(y1)
        z2 = self.intra_video_projector(y2)

        return z1, z2

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
        z1, z2 = self.intra_video_step(batch, batch_idx)
        intra_video_loss = self.intra_video_loss(z1, z2)

        
        losses = {}

        losses['loss'] = intra_video_loss['loss']
        losses['invariance_loss'] = intra_video_loss['invariance_loss']
        losses['variance_loss'] = intra_video_loss['variance_loss']
        losses['covariance_loss'] = intra_video_loss['covariance_loss']

        

        # log results
        self.log_dict(
            {
                "train_loss": losses['loss'],
                "train_invariance_loss": losses['invariance_loss'],
                "train_variance_loss": losses['variance_loss'],
                "train_covariance_loss": losses['covariance_loss'],
            }
        )
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the evaluation cycle.
        For this simple example it's mostly the same as the training loop
        but with a different metric name.
        """
        z1, z2 = self.intra_video_step(batch, batch_idx)
        intra_video_loss = self.intra_video_loss(z1, z2)

        
        losses = {}

        losses['loss'] = intra_video_loss['loss']
        losses['invariance_loss'] = intra_video_loss['invariance_loss']
        losses['variance_loss'] = intra_video_loss['variance_loss']
        losses['covariance_loss'] = intra_video_loss['covariance_loss']

        

        # log results
        self.log_dict(
            {
                "train_loss": losses['loss'],
                "train_invariance_loss": losses['invariance_loss'],
                "train_variance_loss": losses['variance_loss'],
                "train_covariance_loss": losses['covariance_loss'],
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
        if self.optimizer == "lars":
            optimizer = LARS(
                params,
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=self.args.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.optimizer == "adam":
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

