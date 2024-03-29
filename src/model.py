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

from pytorch_lightning.callbacks import ModelSummary

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
        """
        Args:
            args (object): Parser with the configuration arguments.
        """
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.unfreeze_last_layer = False

        # Clean CUDA Cache
        self.clean_cache()

        # Init backbone
        self.video_backbone = self.init_video_backbone()
        self.audio_backbone = self.init_audio_backbone()

        # Init intra mode Projector
        if self.args.intra_mod:
            self.intra_video_projector = self.init_projector(
                args.video_representations_dim,
                args.intra_video_projector
            )

            self.intra_audio_projector = self.init_projector(
                args.audio_representations_dim,
                args.intra_audio_projector
            )

        # Init cross mode Projector
        if self.args.cross_mod:
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
    
    def clean_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def init_video_backbone(self):
        """
        Create video backbone and set the size of video representations.
        """
        if self.args.backbone_video == 'byol_video':
            video_backbone = load_pretrained_video_byol(args=self.args)
            self.args.video_representations_dim = 2048
        
        # Freeze backbone
        video_backbone = self.freeze_model_weights(video_backbone)
        
        # Unfreeze layer to pretrain intra modality
        if self.args.intra_mod:
            video_backbone = self.unfreeze_layer_weights(video_backbone, key="blocks.4")
            
        return video_backbone

    def init_audio_backbone(self):
        """
        Create audio backbone and set the size of audio representations.
        """
        if self.args.backbone_audio == 'byol_audio':
            audio_backbone = load_pretrained_audio_byol(args=self.args)
            self.args.audio_representations_dim = 3072
        
        # Freeze backbone
        audio_backbone = self.freeze_model_weights(audio_backbone)

        # Unfreeze layer to pretrain intra modality
        if self.args.intra_mod:
            audio_backbone = self.unfreeze_layer_weights(audio_backbone, key="fc")
        return audio_backbone
    
    def freeze_model_weights(self, model):
        """
        Freeze the model weights based on the layer names.

        Args:
            model (torch.Module): Input model.
        
        Returns:
            torch.Module: model with all parameters freeze.
        """
        print('Going to apply weight frozen')
        for name, para in model.named_parameters():
            para.requires_grad = False
            
        print('after frozen, require grad parameter names:')
        for name, para in model.named_parameters():
            if para.requires_grad:print(name)
        return model
    
    def unfreeze_layer_weights(self, model, key):
        """
        Unfreeze the model weights based on the layer names.
        
        Args:
            model (torch.Module): Input model.
            key (str): Initial substring of the target layer. 
        Returns:
            torch.Module: model with layer unfreeze.
        """
        print('Going to apply weight unfrozen')
        for name, para in model.named_parameters():
            if name.startswith(key):
                para.requires_grad = True
        
        print('after unfrozen, require grad parameter names:')
        for name, para in model.named_parameters():
            if para.requires_grad:print(name)
        return model

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
        # Representations per modality
        video = batch['video']
        audio = batch['audio']

        video_reps = self.video_representation_step(video)
        audio_reps = self.audio_representation_step(audio)

        # Get losses intra modality
        if self.args.intra_mod:
            intra_video_loss = self.intra_video_step(video_reps)
            intra_audio_loss = self.intra_audio_step(audio_reps)
        else:
            intra_video_loss, intra_audio_loss = (0.0, 0.0)

        # Get losses cross modality
        if  self.args.cross_mod:
            video_rep1, video_rep2 = video_reps
            audio_rep1, audio_rep2 = audio_reps

            cross_video_audio_loss = (
                self.cross_video_audio_step(video_rep1, audio_rep1) * 0.5  +
                self.cross_video_audio_step(video_rep2, audio_rep2) * 0.5
            )
        else:
            cross_video_audio_loss = 0.0

        # Get total weighted loss
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
    
    def video_representation_step(self, samples):
        """
        Calculates the video representations of clips from the same video.

        Args:
            samples (tuple): Random video clips from the same video.

        Returns:
            tuple containing:
                - (video_rep1, video_rep2): Video representations for clip1 and clip2.
        """
        # video: batches of transform views
        video1, video2 = samples

        # video: batches of representations
        if self.trainer.current_epoch < self.args.backbone_freeze_epochs:
            with torch.no_grad():
                video_rep1 = self.video_backbone(video1)
                video_rep2 = self.video_backbone(video2)
        else:
            video_rep1 = self.video_backbone(video1)
            video_rep2 = self.video_backbone(video2)

        return (video_rep1, video_rep2)

    def intra_video_step(self, video_representations):
        """
        Representations are expanded using a projection head. Finally,
        the variance, invariance and covariance loss is calculated.

        Args:
            video_representations (tuple):  Video representations for clip1 and clip2. 

        Returns:
            tuple containing:
                - intra_video_loss: VICReg Loss for video intra modality.
        """
        # video: reprsentations
        video_rep1, video_rep2 = video_representations

        # video: batches of embeddings
        video_emb1 = self.intra_video_projector(video_rep1)
        video_emb2 = self.intra_video_projector(video_rep2)

        # loss intra video modality
        intra_video_loss = self.loss_intra_video(video_emb1, video_emb2)
        return intra_video_loss

    def audio_representation_step(self, samples):
        """
        Calculates the audio representations of clips from the same audio.

        Args:
            samples (tuple): Random audio clips from the same audio.

        Returns:
            tuple containing:
                - (audio_rep1, audio_rep2): Audio representations for clip1 and clip2.
        """
        # audio: batches of transform views
        audio1, audio2 = samples

        # audio: batches of representations
        if self.trainer.current_epoch < self.args.backbone_freeze_epochs:
            with torch.no_grad():
                audio_rep1 = self.audio_backbone(audio1)
                audio_rep2 = self.audio_backbone(audio2)
        else:
            audio_rep1 = self.audio_backbone(audio1)
            audio_rep2 = self.audio_backbone(audio2)

        return (audio_rep1, audio_rep2)

    def intra_audio_step(self, audio_representations):
        """
        Calculates the audio representations of clips from the same audio.
        Such representations are expanded using a projection head. Finally,
        the variance, invariance and covariance loss is calculated.

        Args:
            audio_representations (tuple): Audio representations for clip1 and clip2.

        Returns:
            tuple containing:
                - intra_audio_loss: VICReg Loss for audio intra modality.
        """
        
        # audio: representations
        (audio_rep1, audio_rep2) = audio_representations

        # audio: batches of embeddings
        audio_emb1 = self.intra_audio_projector(audio_rep1)
        audio_emb2 = self.intra_audio_projector(audio_rep2)

        # loss intra audio modality
        intra_audio_loss = self.loss_intra_audio(audio_emb1, audio_emb2)
        return intra_audio_loss

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
        self.log('train/loss', losses['loss'], on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
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
        """
        Args:
            args (object): Parser with the configuration arguments.
        """
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        # Init video backbone
        self.backbone = self.init_backbone()
        self.crossmod_proj = self.init_crossmod_proj()
        self.linear_layer = self.init_linear_layer()

        # metrics
        self.accuracy = Accuracy(task="multiclass", num_classes=self.args.num_classes)

    def init_backbone(self):
        """
        Load pretrain Multi-Mode VICRec model or original BYOL model. Selects the modality backbone 
        (e.g. audio or video) that will be used during the evaluation protocol.

        Returns:
            nn.Module: Modality backbone.
        """

        model = torch.load(self.args.checkpoint_path)

        # Selecting Backbone modality for evaluation protocol.
        if self.args.eval_data_modality == 'video':
            backbone = model.video_backbone
            self.args.representations_dim = model.args.video_representations_dim 

        elif self.args.eval_data_modality == 'audio':
            backbone = model.audio_backbone
            self.args.representations_dim = model.args.audio_representations_dim 
        
        return backbone
    
    def init_crossmod_proj(self):
        """
        Load pretrain Cross-Mode projector from the VICRec model. Selects the modality backbone 
        (e.g. audio or video) that will be used during the evaluation protocol.

        Returns:
            nn.Module: Cross Modality projector.
        """
        # Selecting Backbone modality for evaluation protocol.
        if self.args.eval_data_modality == 'video':

            if self.args.crossmod_proj_eval:
                model = torch.load(self.args.checkpoint_path)
                cross_mod_projector = model.cross_video_to_audio_projector
                self.args.projector_dim = int(model.args.cross_audio_to_video_projector.split("-")[-1])
            else:
                cross_mod_projector = None
        
        if self.args.eval_data_modality == 'audio':
            
            if self.args.crossmod_proj_eval:
                model = torch.load(self.args.checkpoint_path)
                cross_mod_projector = model.cross_audio_to_video_projector
                self.args.projector_dim = int(model.args.cross_audio_to_video_projector.split("-")[-1])
            else:
                cross_mod_projector = None
        return cross_mod_projector

    def init_linear_layer(self):
        """
        Layer for linear evaluation. 
        """
        if self.args.crossmod_proj_eval:
            input_dim = self.args.representations_dim + self.args.projector_dim 
        else:
            input_dim = self.args.representations_dim
        
        linear_layer =  nn.Sequential(
            nn.Dropout(p=self.args.eval_dropout_p), 
            nn.Linear(input_dim, self.args.num_classes, bias=True)
        )
        return linear_layer
 
    def training_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the training epoch.
        It must return a loss that is used for loss.backwards() internally.
        """
        loss, acc = self.shared_step(batch)
        
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the validation epoch.
        """
        loss, acc = self.shared_step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the test epoch.
        """
        loss, acc = self.shared_step(batch)

        self.log("test/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, sync_dist=True)

    def shared_step(self, batch):
        """
        Linear projections of representations are obtained and cross entropy
        loss is calcualted with respect to the labels. The backbone weights
        are only updated if the evaluation protocol is `finetune`.
        
        Args:
            batch: {
                "video":<video_tensor> (Optional),
                "audio":<audio_tensor> (Optional),
                "label": <label>
            }

        Returns:
            tuple: (cross entropy loss, accuracy metric)
        """
        # Get data specific to modality
        x = batch[self.args.eval_data_modality]

        # Get labels
        labels = batch['label']

        # Representations from pretrain backbone
        if self.args.eval_protocol == 'linear':
            with torch.no_grad():
                representations = self.backbone(x)

                # If cross mod projection is enable
                if self.args.crossmod_proj_eval:
                    crossmod_emb = self.crossmod_proj(representations)
                    representations = torch.cat((representations, crossmod_emb), dim=1)

        elif self.args.eval_protocol == 'finetune':
            representations = self.backbone(x)

            # If cross mod projection is enable
            if self.args.crossmod_proj_eval:
                crossmod_emb = self.crossmod_proj(representations)
                representations = torch.cat((representations, crossmod_emb), dim=1)


        # Linear projection
        y_hat = self.linear_layer(representations)

        # Loss
        loss = F.cross_entropy(y_hat, labels)
        acc = self.accuracy(F.softmax(y_hat, dim=-1), labels)
    
        return loss, acc
  
    def configure_optimizers(self):
        """
        We use the SGD optimizer with multi-step learning rate scheduler.
        scheduler.
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args.eval_learning_rate,
            momentum=0.9,
            weight_decay=self.args.eval_weight_decay,
        )

        # set scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.args.eval_decay_epochs, gamma=self.args.eval_gamma)

        return [optimizer], [scheduler]

