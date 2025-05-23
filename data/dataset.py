import glob
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from random import randrange
from torchvision.io import decode_image

class StyleTransferDataset(Dataset):
    """
    Dataset class for style transfer training.
    """
    def __init__(
        self,
        style_dir,
        content_dir,
        num_styles_per_image=100
    ):
        """
        Init dataset with style/content paths and sampling setup.
        """
        super().__init__()
        self.num_styles_per_image = num_styles_per_image
        self.style_paths = {}
        self.rng = np.random.default_rng(42)

        # Load style image paths
        style_paths = glob.glob(os.path.join(style_dir, '*.jpg'))
        self.total_style_images = len(style_paths)
        for i, filepath in enumerate(style_paths):
            self.style_paths[i] = filepath

        # Load content image paths
        self.content_paths = {}
        content_paths = glob.glob(os.path.join(content_dir, '*.jpg'))
        self.total_content_images = len(content_paths)
        for i, filepath in enumerate(content_paths):
            self.content_paths[i] = filepath

        # Prepare indices for sampling
        self.all_inds = self._build_indices()

    def _build_indices(self):
        """
        Build repeated and shuffled content indices.
        """
        base_inds = np.arange(self.total_content_images)
        all_inds = np.repeat(base_inds, self.num_styles_per_image)
        self.rng.shuffle(all_inds)
        return all_inds

    def __len__(self):
        """
        Return dataset length.
        """
        return len(self.all_inds)

    def __getitem__(self, idx):
        """
        Return a sample with style and random content image.
        """
        style = decode_image(self.style_paths[self.all_inds[idx]]).to(torch.float32).div(255.0)
        content = decode_image(self.content_paths[randrange(self.total_content_images)]).to(torch.float32).div(255.0)
        
        # Downsample style
        style = F.interpolate(
            style.unsqueeze(0),
            (256, 256),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        
        # Upsample content
        content = F.interpolate(
            content.unsqueeze(0),
            (256, 256),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        sample = {
            "style_image": style,
            "content_image": content
        }
        return sample
