import os
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from torch import nn
from torch.nn import functional as F

import torchvision

import pytorch_lightning

from torchmetrics import Accuracy

from optim import LARS
from optim import exclude_from_wt_decay
from optim import CosineAnnealingWarmupRestarts

from loss import VICRegLoss
from backbones.byol_video.model import load_pretrained_video_byol
from backbones.byol_audio.model import load_pretrained_audio_byol


class MultiModVICRegModule(pytorch_lightning.LightningModule):
    """
    PyTorch Lightning implementation of Multi-Modal VICReg.
    """
    def __init__(self, args):

        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.warmup_stage = True

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

        self.num_features_intra_video = int(args.intra_video_projector.split("-")[-1])
        self.num_features_intra_audio = int(args.intra_audio_projector.split("-")[-1])
        self.num_features_cross_video_to_audio= int(args.cross_video_to_audio_projector.split("-")[-1])

        # Init loss
        self.loss_intra_video = VICRegLoss(
            args.invariance_coeff,
            args.variance_coeff,
            args.covariance_coeff,
            args.batch_size,
            args.num_nodes,
            args.devices,
            self.num_features_intra_video
            )
        
        self.loss_intra_audio  = VICRegLoss(
            args.invariance_coeff,
            args.variance_coeff,
            args.covariance_coeff,
            args.batch_size,
            args.num_nodes,
            args.devices,
            self.num_features_intra_audio 
            )

        self.loss_cross_video_audio = VICRegLoss(
            args.invariance_coeff,
            args.variance_coeff,
            args.covariance_coeff,
            args.batch_size,
            args.num_nodes,
            args.devices,
            self.num_features_cross_video_to_audio
            )
    
    def init_video_backbone(self):
        """
        Create video backbone 
        """
        if self.args.backbone_video == 'byol_video':
            video_backbone = load_pretrained_video_byol(args=self.args)
            self.args.video_representations_dim = 2048

        return video_backbone

    def init_audio_backbone(self):
        """
        Create audio backbone.
        """
        if self.args.backbone_audio == 'byol_audio':
            audio_backbone = load_pretrained_audio_byol(args=self.args)
            self.args.audio_representations_dim = 3072
            
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
            layers.append(nn.Linear(f[i], f[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward defines the prediction/inference actions.
        """
        video, audio = x['video'], x['audio']
        return (self.video_backbone(video), self.audio_backbone(audio))

    def share_step(self, batch, batch_idx):
        """
        Calculates total, intra modality and cross modality losses.

        Returns:
            dict: Summary of losses.
        """
        video = batch['video']
        audio = batch['audio']
        
        intra_video_loss, video_reps = self.intra_video_step(video)
        intra_audio_loss, audio_reps = self.intra_audio_step(audio)

        video_rep1, video_rep2 = video_reps
        audio_rep1, audio_rep2 = audio_reps

        cross_video_audio_loss = (
            self.cross_video_audio_step(video_rep1, audio_rep1) * 0.5  +
            self.cross_video_audio_step(video_rep2, audio_rep2) * 0.5
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
        if self.warmup_stage:
            with torch.no_grad():
                video_rep1 = self.video_backbone(video1)
                video_rep2 = self.video_backbone(video2)
        else:
            video_rep1 = self.video_backbone(video1)
            video_rep2 = self.video_backbone(video2)

        # video: batches of embeddings
        video_emb1 = self.intra_video_projector(video_rep1)
        video_emb2 = self.intra_video_projector(video_rep2)

        # loss intra video modality
        intra_video_loss = self.loss_intra_video(video_emb1, video_emb2)
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
        if self.warmup_stage:
            with torch.no_grad():
                audio_rep1 = self.audio_backbone(audio1)
                audio_rep2 = self.audio_backbone(audio2)
        else:
            audio_rep1 = self.audio_backbone(audio1)
            audio_rep2 = self.audio_backbone(audio2)

        # audio: batches of embeddings
        audio_emb1 = self.intra_audio_projector(audio_rep1)
        audio_emb2 = self.intra_audio_projector(audio_rep2)

        # loss intra audio modality
        intra_audio_loss = self.loss_intra_audio(audio_emb1, audio_emb2)
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

        cross_video_audio_loss = self.loss_cross_video_audio(cross_video_audio_emb, cross_audio_video_emb)
        return cross_video_audio_loss
    
    def on_train_epoch_start(self):
        """
        For distributed training we need to set the datasets video sampler epoch so
        that shuffling is done correctly
        """
        epoch = self.trainer.current_epoch
        if self.args.num_nodes * self.args.devices > 1:
            self.trainer.datamodule.train_dataset.dataset.video_sampler.set_epoch(epoch)
        
        if epoch >= self.args.warmup_epochs:
            self.warmup_stage = False

    def training_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the training epoch.
        It must return a loss that is used for loss.backwards() internally.

        batch: {
            "video":[<video_tensor>, <video_tensor_prime>],
            "audio": [<audio_tensor>, <audio_tensor_prime>]
        }

        - "video" is a Tensor of shape (batch, channels, time, height, Width)
        - "audio" is a Tensor of shape (batch, channels, time, 1, frequency)
        
        The PyTorchVideo models and transforms expect the same input shapes and
        dictionary structure making this function just a matter of unwrapping
        the dict and feeding it through the model/loss.
        """
        losses = self.share_step(batch, batch_idx)

        # log results
        self.log('train/loss', losses['loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/intra_video_loss', losses['intra_video_loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/intra_audio_loss', losses['intra_audio_loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/cross_video_audio_loss', losses['cross_video_audio_loss'], on_step=False, on_epoch=True, sync_dist=True)

        return losses['loss']

    def configure_optimizers(self):
        """
        We use the LARS/Adam optimizer with per step cosine annealing
        scheduler.
        """
        
        args = self.args

        if args.exclude_bn_bias:
            params = exclude_from_wt_decay(self.named_parameters(), weight_decay=args.weight_decay)
        else:
            params = self.parameters()

        # Optimizer
        if self.args.optimizer == "lars":
            optimizer = LARS(
                params,
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )

        # Scheduler
        args.train_iters_per_epoch = args.num_train_samples // (args.batch_size * args.devices * args.num_nodes)
        args.warmup_steps = args.warmup_epochs * args.train_iters_per_epoch
        args.total_steps = args.max_epochs * args.train_iters_per_epoch

        scheduler = {
            "scheduler": CosineAnnealingWarmupRestarts(
                optimizer,
                warmup_steps=args.warmup_steps,
                first_cycle_steps=args.total_steps,
                min_lr=0.001,
                max_lr=args.learning_rate,
                cycle_mult=1.0
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


class EvaluatorModule(pytorch_lightning.LightningModule):
    """
    PyTorch Lightning implementation of Multi-Modal VICReg.
    """
    def __init__(self, args):

        super().__init__()
        self.save_hyperparameters()
        self.args = args

        # Init video backbone
        self.backbone = self.init_backbone()
        self.linear_layer = self.init_linear_layer()

        # metrics
        self.accuracy = Accuracy(task="multiclass", num_classes=self.args.num_classes)

    def init_backbone(self):
        """
        Create video backbone and replace  the head linear projection
        with a Identity function.
        """
        model = MultiModVICRegModule(self.args, None)

        if torch.cuda.is_available():
            self.args.device_type = torch.device("cuda")
        else:
            self.args.device_type = torch.device("cpu")

        checkpoint = torch.load(self.args.checkpoint_path, map_location=self.args.device_type)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.args.device_type)

        if self.args.eval_protocol == 'linear':
                model.freeze()
        
        if self.args.eval_data_modality == 'video':
            backbone = model.video_backbone
            self.args.representations_dim = model.args.video_representations_dim 

        elif self.args.eval_data_modality == 'audio':
            backbone = model.audio_backbone
            self.args.representations_dim = model.args.audio_representations_dim 
        
        return backbone

    def init_linear_layer(self):
        linear_layer =  nn.Sequential(
            nn.Dropout(p=self.args.eval_dropout_p), 
            nn.Linear(self.args.representations_dim, self.args.num_classes, bias=True)
        )
        return linear_layer

    def on_train_epoch_start(self):
        if self.args.eval_protocol == 'linear':
            self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)
        
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, sync_dist=True)

    def shared_step(self, batch):
        # Get data specific to modality
        x = batch[self.args.eval_data_modality]

        # Representations from pretrain backbone
        if self.args.eval_protocol == 'linear':
            with torch.no_grad():
                representations = self.backbone(x)

        elif self.args.eval_protocol == 'finetune':
            representations = self.backbone(x)

        # Linear projection
        y_hat = self.linear_layer(representations)

        # Encode Labels
        if self.args.eval_dataset == 'ucf101':
            label_string = [ video .split('_')[1] for video in batch['video_name']]
            labels = torch.tensor([self.args.dict_labels[l] for l in label_string])
            labels = labels.to(device=self.args.device_type)
        
        elif self.args.eval_dataset == 'hmdb51':
            label_string = batch['label']
            labels = torch.tensor([self.args.dict_labels[l] for l in label_string])
            labels = labels.to(device=self.args.device_type)

        elif self.args.eval_dataset == 'kinetics400':
            labels = batch['label']

        # Loss
        loss = F.cross_entropy(y_hat, labels)
        acc = self.accuracy(F.softmax(y_hat, dim=-1), labels)
    
        return loss, acc
  
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args.eval_learning_rate,
            momentum=0.9,
            weight_decay=self.args.eval_weight_decay,
        )

        # set scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.args.eval_decay_epochs, gamma=self.args.eval_gamma)

        return [optimizer], [scheduler]

