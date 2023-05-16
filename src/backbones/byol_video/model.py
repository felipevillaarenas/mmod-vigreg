import os
import json
from pathlib import Path
from typing import List
import logging

import torch
import torch.nn as nn

from pytorchvideo.models.resnet import create_resnet

def load_pretrained_weights(model, args, strict=True):
    """Parse the values from the pretrain weights to the model.

    Args:
        model (nn.Module): Pytorch model.  
        args (object): Parser with the configuration arguments.
        strict (bool, optional): Dict state strict match. Defaults to True.

    Returns:
        list: Sorted list of weights keys.
    """
    # Filter relevant layer of pretained BYOL model
    prefix = Path(__file__).parent
    map_path = str(prefix.joinpath('map_weights_byol_video.json'))
    map_file = open(map_path)
    map_weights_byol_video = json.load(map_file)

    # Loading pretrain video BYOL weights for Slow R50 model 
    pathname = str(Path(args.path_pretrained_backbone_weights).joinpath('video/video_byol_weights.pyth'))
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        pretrained_weights = torch.load(pathname, map_location='cuda')
    else:
        pretrained_weights = torch.load(pathname, map_location='cpu')

    # Map pretrained weights to model
    pretrained_state_dict =  pretrained_weights['model_state']
    model_state_dict = model.state_dict()

    pretrained_state_dict_filtered = dict()
    old_keys = list(pretrained_state_dict.keys())
    root_keys = list(map_weights_byol_video.keys())
    for key_old in old_keys:
        for root_key in root_keys:
            if key_old.startswith(root_key):
                pretrained_state_dict_filtered[key_old] = pretrained_state_dict[key_old]
    
    # Creating new model state with matching keys
    weights = dict()
    if len(pretrained_state_dict_filtered)==len(model_state_dict):
        keys_model_state = list(model_state_dict.keys())
        for i, key in enumerate(list(pretrained_state_dict_filtered.keys())):
            weights[keys_model_state[i]] = pretrained_state_dict_filtered[key]


    # Udate pretarin BYOL weights in model
    logging.info(f' using network pretrained weight: {Path(pathname).name}')
    logging.info(str(model.load_state_dict(weights, strict=strict)))

    # Free Cache
    if torch.cuda.is_available():
        del pretrained_weights
        torch.cuda.empty_cache()
    return sorted(list(weights.keys()))

def load_pretrained_video_byol(args):
    """Create video backbone and load pretrained weights.

    Args:
        args (object): Parser with the configuration arguments.
    
    Returns:
        nn.Module: backbone model with updated weights.
    """
    # Load Slow R50 model from torch Hub
    video_backbone = create_resnet(
        stem_conv_kernel_size=(1, 7, 7),
        head_pool_kernel_size=(8, 7, 7),
        model_depth=50
    )
    video_backbone._modules['blocks'][-1].dropout = nn.Identity()
    video_backbone._modules['blocks'][-1].proj = nn.Identity()
    load_pretrained_weights(video_backbone, args, strict=True)
    return video_backbone

