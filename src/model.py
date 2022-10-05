import pytorch_lightning
import torch
import torch.nn.functional as F
from pl_bolts.optimizers.lars import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pytorchvideo.models.resnet import create_resnet, create_acoustic_resnet


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
        return self.model(x)

    def vigreg_loss(self, z1, z2):
        # invariance Loss
        invariance_loss = F.mse_loss(z1, z2)

        # Share operation for Variance and Covariance
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        # Variance Loss
        std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
        std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
        variance_loss_z1 = torch.mean(F.relu(1 - std_z1)) / 2
        variance_loss_z2 = torch.mean(F.relu(1 - std_z2)) / 2
        variance_loss = variance_loss_z1 + variance_loss_z2

        # Covariance Loss
        cov_z1 = (z1.T @ z1) / (z1.shape[0] - 1)
        cov_z2 = (z2.T @ z2) / (z2.shape[0] - 1)
        covariance_loss_z1 = self.off_diagonal(cov_z1).pow_(2).sum()
        covariance_loss_z2 = self.off_diagonal(cov_z2).pow_(2).sum()
        covariance_loss_z1 = covariance_loss_z1.div(self.args.num_features_expander)
        covariance_loss_z2 = covariance_loss_z2.div(self.args.num_features_expander)
        covariance_loss = covariance_loss_z1 + covariance_loss_z2

        # Loss function is a weighted average of the loss terms
        loss = (
            self.args.invariance_coeff * invariance_loss
            + self.args.variance_coeff * variance_loss
            + self.args.covariance_coeff * covariance_loss
        )

        return {
            'loss': loss,
            'invariance_loss': invariance_loss,
            'variance_loss': variance_loss,
            'covariance_loss': covariance_loss
        }

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def shared_step(self, batch, batch_idx):
        # x1, x2: batches of transform views
        (x1, x2, _), _ = batch

        # y1, y2: batches of representations
        y1 = self(x1)
        y2 = self(x2)

        # z1, z2: batches of embeddings
        z1 = self.projector(y1)
        z2 = self.projector(y2)

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
        z1, z2 = self.shared_step(batch, batch_idx)
        vigreg_loss = self.vigreg_loss(z1, z2)

        # log results
        self.log_dict(
            {
                "train_loss": vigreg_loss['loss'],
                "train_invariance_loss": vigreg_loss['invariance_loss'],
                "train_variance_loss": vigreg_loss['variance_loss'],
                "train_covariance_loss": vigreg_loss['covariance_loss'],
            }
        )
        return vigreg_loss['loss']

    def validation_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the evaluation cycle.
        For this simple example it's mostly the same as the training loop
        but with a different metric name.
        """
        z1, z2 = self.shared_step(batch, batch_idx)
        vigreg_loss = self.vigreg_loss(z1, z2)

        # log results
        self.log_dict(
            {
                "val_loss": vigreg_loss['loss'],
                "val_invariance_loss": vigreg_loss['invariance_loss'],
                "val_variance_loss": vigreg_loss['variance_loss'],
                "val_covariance_loss": vigreg_loss['covariance_loss'],
            }
        )
        return vigreg_loss['loss']

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

