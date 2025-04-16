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
        self.avg_pool = nn.AvgPool2d(kernel_size=n)
        
    def forward(self, patch_embs):
        batch_size, num_patches, embed_dim = patch_embs.shape
        
        h = w = int(np.sqrt(num_patches))
        patch_embs_2d = patch_embs.transpose(1, 2).view(batch_size, embed_dim, h, w)
        
        P_L = self.F_pos(self.avg_pool(patch_embs_2d))
        
        P_L_resized = F.interpolate(P_L, size=(h, w), mode='bilinear', align_corners=False)
        
        P_CA = P_L_resized.flatten(2).transpose(1, 2)
        
        final_embeddings = patch_embs + P_CA
        
        return final_embeddings