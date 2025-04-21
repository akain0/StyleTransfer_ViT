import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CAPE(nn.Module):
    '''
    Content-Aware Positional Encoding class for content-aware, 'understood' semantics of image content.
    Combined with content patches (down-projected patches of content image) to be fed into transformer encoder layer.
    '''
    def __init__(self, embed_dim=512, n=18, s=1):
        super(CAPE, self).__init__()
        self.embed_dim = embed_dim  
        self.n = n                  
        self.s = s                  
        
        self.F_pos = nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(n)
        
    def forward(self, patch_embs):
        batch_size, embed_dim, h, w = patch_embs.shape
        
        pos_embed = self.F_pos(self.avg_pool(patch_embs))
        pos_embed = F.interpolate(pos_embed, size=(h, w), mode='bilinear', align_corners=False)
        
        embeddings = patch_embs + pos_embed
        return embeddings
