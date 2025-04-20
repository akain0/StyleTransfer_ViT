import torch
import torch.nn as nn
from torchvision import models

class VGGFeatureExtractor(nn.Module):
    """
    Wrap VGG19 to grab fixed activations for style/content.
    
    Runs input through VGG19 features and returns a dict of
    layer activations keyed by intuitive names.
    """
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.layer_idx = {
            "state_1": 1,
            "state_2": 6,
            "state_3": 11,
            "state_4": 20,
            "state_5": 22,
            "state_6": 29
        }
        self.idx_to_name = {idx: name for name, idx in self.layer_idx.items()}
        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x):
        """
        Collect activations at predefined layers.
        """
        feats = {}
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            name = self.idx_to_name.get(idx)
            if name is not None:
                feats[name] = x
        return feats
