"""Model Script Used By train.py and predict.py"""
import torch
from torch import nn

from networks.eccv16 import eccv16

def get_model(model_name: str) -> nn.Module:
    """Get the nn.Module based on input"""
    if model_name == 'eccv16':
        return eccv16(pretrained=False)
    if model_name == 'eccv16_pretrained':
        return eccv16(pretrained=True)
    raise ValueError('Unknown model')
