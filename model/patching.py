import torch.nn as nn
import torch.nn.functional as F

class ContentPatching(nn.Module):
    '''
    Content Patching class for patching content image into patches (i,j)
    Content images will be fed in as batch x 3 x 150 x 150
    '''
    def __init__(
        self,
        in_channels = 3,
        embed_dim = 512,
        patch_size = 8
    ):
        super(ContentPatching, self).__init__()
        self.patch_embed = nn.Conv2d(
            in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, src_img):
        patches = self.patch_embed(src_img)
        patches = patches.flatten(2)
        patch_embeddings = patches.transpose(1,2)
        return patch_embeddings
        
class StylePatching(nn.Module):
    '''
    Style Patching class for patching style image into patches (i,j)
    Style images will be fed in as batch x 3 x 512 x 512
    '''
    def __init__(
        self,
        in_channels = 3,
        embed_dim = 512,
        patch_size = 8
    ):
        super(StylePatching, self).__init__()
        self.patch_embed = nn.Conv2d(
            in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, src_img):
        patches = self.patch_embed(src_img)
        patches = patches.flatten(2)
        patch_embeddings = patches.transpose(1,2)
        return patch_embeddings

        
