import torch
import torch.nn as nn
import numpy as np

class TransformerDecoder(nn.Module):
    '''
    Transformer Decoder class to be leveraged for decoding the target token sequences
    based on encoded memory representations from the encoder.
    '''
    def __init__(
        self,
        d_model = 512,
        nhead = 8,
        dim_feedforward = 2048,
        dropout = 0.1,
        n_layers = 6
    ):
        super(TransformerDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=n_layers
        )

    def forward(
        self,
        target_tokens,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None
    ):
        out = self.transformer_decoder(
            tgt=target_tokens,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return out
    

class CNNDecoder(nn.Module):
    def __init__(self, embed_dim):
        """
        Upsampling decoder to output the stylized image.
        The input to this layer are the transformer's outputs
        Note:
            The transformer output has a shape of (batch, (H*W)/64, embed_dim).
            This implies that when the features are reshaped into a grid,
            the grid size will be (H/8) x (W/8), since (H*W/64) = (H/8)*(W/8).
            To reshape the content, we will utilize the fact that the content is
            always square.
        """
        super(CNNDecoder, self).__init__()        
        # First layer: embed_dim --> embed_dim//2; upscales by 2.
        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(embed_dim, embed_dim//2, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(embed_dim//2, embed_dim//2, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(embed_dim//2, embed_dim//2, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(embed_dim//2, embed_dim//2, (3, 3)),
            nn.ReLU()
        )
        
        # Second layer: embed_dim//2 --> embed_dim//4; upscales by 2.
        self.layer2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(embed_dim//2, embed_dim//4, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(embed_dim//4, embed_dim//4, (3, 3)),
            nn.ReLU()
        )
        
        # Third layer: embed_dim//4 --> 3; upscales by 2.
        self.layer3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(embed_dim//4, embed_dim//8, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(embed_dim//8, embed_dim//8, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(embed_dim//8, 3, (3, 3))
        )

    def forward(self, x):
        """
        Forward pass of the decoder.
        """
        batch_size, num_tokens, channels = x.shape
        # Calculate grid dimensions (m = 8)
        grid_w = grid_h = int(np.sqrt(num_tokens))
        
        # Reshape flattened tokens
        x = x.permute(0, 2, 1).reshape(batch_size, channels, grid_h, grid_w)
        
        # Pass through the three-layer CNN decoder.
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        return x    # Shape: (batch, 3, H, W)
