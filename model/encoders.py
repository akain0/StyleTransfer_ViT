import torch.nn as nn

class ContentTransformerEncoder(nn.Module):
    '''
    Content Transformer Encoder class to be leveraged for encoding the
    added CAPE + content patch sequences into contextualized 
    representations that get mixed with style representations.
    '''
    def __init__(
        self,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        n_layers=6
    ):
        super(ContentTransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
    def forward(self, content_patches_plus_cape):
        out = self.transformer_encoder(content_patches_plus_cape)
        return out
    
class StyleTransformerEncoder(nn.Module):
    '''
    Style Transformer Encoder class to be leveraged for encoding the
    style image's patch sequences into contextualized representations
    for styling content image representations.
    '''
    def __init__(
        self,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        n_layers=6
    ):
        super(StyleTransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
    def forward(self, style_patches):
        out = self.transformer_encoder(style_patches)
        return out
