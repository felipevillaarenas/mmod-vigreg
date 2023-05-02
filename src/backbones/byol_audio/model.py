"""Model BYOL for Audio.

Reference:
- Niizumi, Daisuke and Takeuchi, Daiki and Ohishi, Yasunori and Harada, Noboru and Kashino, Kunio, 
  “BYOL for Audio: Exploring Pre-trained General-purpose Audio Representations,” TASLP 2022
  ttps://arxiv.org/pdf/2204.07402.pdf
"""

import logging
from pathlib import Path
import torch
from torch import nn


def load_pretrained_weights(model, args, model_key='model', strict=True):
    # Loading pretrain Audio BYOL weights
    pathname = str(Path(args.path_pretrained_backbone_weights).joinpath('audio/audio_byol_weights.pyth'))

    if torch.cuda.is_available():
        state_dict = torch.load(pathname, map_location='cuda')
    else:
        state_dict = torch.load(pathname, map_location='cpu')

    # Mapping weights
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if 'model' in state_dict:
        state_dict = state_dict['model']
    children = sorted([n + '.' for n, _ in model.named_children()])

    weights = {}
    for k in state_dict:
        weights[k[len(model_key)+1:] if k.startswith(model_key+'.') else k] = state_dict[k]
    state_dict = weights

    def find_model_prm(k):
        for name in children:
            if name in k: # ex) "conv_block1" in "model.conv_block1.conv1.weight"
                return k
        return None

    weights = {}
    for k in state_dict:
        if find_model_prm(k) is None: continue
        weights[k] = state_dict[k]

    logging.info(f' using network pretrained weight: {Path(pathname).name}')
    logging.info(str(model.load_state_dict(weights, strict=strict)))
    return sorted(list(weights.keys()))

def load_pretrained_audio_byol(args):
    """
    Create video backbone and load pretrained weights
    """
    audio_backbone = AudioBYOL(n_mels=64, d=3072)
    load_pretrained_weights(audio_backbone, args=args)
    return audio_backbone

def mean_max_pooling(frame_embeddings):
    assert len(frame_embeddings.shape) == 3 # Batch,Time,Dimension
    (x1, _) = torch.max(frame_embeddings, dim=1)
    x2 = torch.mean(frame_embeddings, dim=1)
    x = x1 + x2
    return x


class AudioBYOLEncoder(nn.Module):
    """General Audio Feature Encoder Network"""

    def __init__(self, n_mels, d=3072, base_d=64, mlp_hidden_d=2048, conv_layers=2, stack=True):
        super().__init__()
        convs = [
            nn.Conv2d(1, base_d, 3, stride=1, padding=1),
            nn.BatchNorm2d(base_d),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        ]
        for c in range(1, conv_layers):
            convs.extend([
                nn.Conv2d(base_d, base_d, 3, stride=1, padding=1),
                nn.BatchNorm2d(base_d),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            ])
        self.features = nn.Sequential(*convs)
        self.conv_d = base_d * (n_mels//(2**conv_layers))
        self.fc = nn.Sequential(
            nn.Linear(self.conv_d, mlp_hidden_d),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(mlp_hidden_d, d - self.conv_d),
            nn.ReLU(),
        )
        self.stack = stack

    def forward(self, x):
        x = self.features(x)       # (batch, ch, mel, time)
        x = x.permute(0, 3, 2, 1)  # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D)) # (batch, time, mel*ch)
        x_fc = self.fc(x)
        x = torch.hstack([x.transpose(1,2), x_fc.transpose(1,2)]).transpose(1,2) if self.stack else x_fc
        return x

class AudioBYOL(AudioBYOLEncoder):
    def __init__(self, n_mels, d=3072, mlp_hidden_d=2048):
        super().__init__(n_mels=n_mels, d=d, mlp_hidden_d=mlp_hidden_d)

    def forward(self, x):
        x = super().forward(x)
        x = mean_max_pooling(x)
        return x