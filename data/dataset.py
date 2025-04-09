import glob
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from random import randrange
from torchvision.io import decode_image


# Dataset Class
class StyleTransferDataset(Dataset):
    def __init__(
        self,
        style_path,
        content_path,
        num_styles_per_image=100
    ):
        super().__init__()
        self.num_styles_per_image = num_styles_per_image
        self.style_paths = {}

        #   If need to extract information from file name
        #   import re
        #   pattern = re.compile(r'^(.*?)_(.*?)_(\d+)\.jpg$')  
        #   e.g. match = pattern.match(filename); artist_name = match.group(3)

        ##   STYLE FILEPATHS FOR RANDOM SAMPLING   ##
        # Use glob to get all .jpg files in the folder
        self.style_paths = {}   # storing style image paths
        style_paths = glob.glob(os.path.join(style_path, '*.jpg'))
        self.total_style_images = len(style_paths)
        for i, filepath in enumerate(style_paths):
            self.style_paths[i] = filepath
            i += 1

        ##   CONTENT FILEPATHS  ##
        # Use glob to get all .jpg files in the folder
        self.content_paths = {}   # storing style image paths
        content_paths = glob.glob(os.path.join(content_path, '*.jpg'))
        self.total_content_images = len(content_paths)
        for i, filepath in enumerate(content_paths):
            self.content_paths[i] = filepath
        
        # Construct indices for content images (repeating self.num_styles_per_image times)
        self.all_inds = self._build_indices()
    
    def _build_indices(self):
        # Construct indices for the dataset class
        base_inds = np.arange(self.total_content_images)
        all_inds = np.repeat(base_inds, self.num_styles_per_image)
        np.random.shuffle(all_inds)   # shuffles in-place
        return all_inds

    def __len__(self):
        # Return the total number of samples
        return len(self.all_inds)

    def __getitem__(self, idx):
        # Retrieve a sample by index
        style = decode_image(self.style_paths[self.all_inds[idx]])
        content = decode_image(self.content_paths[randrange(self.total_content_images)])
        sample = {
            "style_image": style.to(torch.float32),
            "content_image": content.to(torch.float32)
        }
        return sample
