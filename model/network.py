import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from model.patching import ContentPatching, StylePatching
from model.cape import CAPE
from model.encoder import ContentTransformerEncoder, StyleTransformerEncoder
from model.decoder import TransformerDecoder, CNNDecoder

class StyTR2(pl.LightningModule):
    """
    StyTR-2 model with perceptual losses.
    
    Combines content/style patching, transformer encoders/decoder,
    CNN upsampling, and VGG-based content & style losses.
    """
    def __init__(
        self,
        # Universal
        d_model=512,
        img_height=224,
        img_width=224,
        # Patchig
        n=18,
        s=1,
        # Content Encoder
        patch_size_content=8,
        n_head_content=8,
        dim_feedforward_content=2048,
        dropout_content=0.1,
        n_layers_content=6,
        # Style Encoder
        patch_size_style=32,
        n_head_style=8,
        dim_feedforward_style=2048,
        dropout_style=0.1,
        n_layers_style=6,
        # Decoder
        n_head_dec=8,
        dim_feedforward_dec=2048,
        dropout_dec=0.1,
        n_layers_dec=6,
        # Training
        lr=1e-3,
        style_loss_weight=1.0,
        content_loss_weight=1.0,
        lr_patience=5,
        lr_decay=0.1,
        betas=(0.9, 0.999)
    ):
        super().__init__()
        # Patching
        self.content_patcher = ContentPatching(3, d_model, patch_size_content)
        self.style_patcher = StylePatching(3, d_model, patch_size_style)
        self.cape = CAPE(embed_dim=d_model, n=n, s=s)

        # Encoding
        self.content_encoder = ContentTransformerEncoder(
            d_model=d_model,
            nhead=n_head_content,
            dim_feedforward=dim_feedforward_content,
            dropout=dropout_content,
            n_layers=n_layers_content
        )
        self.style_encoder = StyleTransformerEncoder(
            d_model=d_model,
            nhead=n_head_style,
            dim_feedforward=dim_feedforward_style,
            dropout=dropout_style,
            n_layers=n_layers_style
        )

        # Decoding
        self.decoder = TransformerDecoder(
            d_model=d_model,
            nhead=n_head_dec,
            dim_feedforward=dim_feedforward_dec,
            dropout=dropout_dec,
            n_layers=n_layers_dec
        )
        self.cnn_decoder = CNNDecoder(
            embed_dim=d_model,
            img_height=img_height,
            img_width=img_width
        )

        # VGG
        self.vgg_extractor = VGGFeatureExtractor()
        self.content_layer = "content_1"
        self.style_layers = ["style_1", "style_2", "style_3", "style_4", "style_5"]

        # Training params
        self.img_height = img_height
        self.img_width = img_width
        self.style_loss_weight = style_loss_weight
        self.content_loss_weight = content_loss_weight
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_decay = lr_decay
        self.betas = betas

        # Test outputs
        self._test_outputs = []
        self.test_results = None

    def forward(self, style, content):
        """
        Stylize content using style.
        """
        # Standardize image dimensions
        style = F.interpolate(
            style,
            (self.img_height, self.img_width),
            mode="bicubic",
            align_corners=False
        )
        content = F.interpolate(
            content,
            (self.img_height, self.img_width),
            mode="bicubic",
            align_corners=False
        )
        
        # Patching
        c_patches = self.content_patcher(content)
        s_patches = self.style_patcher(style)
        c_emb = self.cape(c_patches)

        # Encoder
        c_enc = self.content_encoder(c_emb)
        s_enc = self.style_encoder(s_patches)

        # Decoder
        tokens = self.decoder(target_tokens=c_enc, memory=s_enc)
        stylized = self.cnn_decoder(tokens)
        return style, content, stylized
    
    def gram_matrix(self, feat):
        """
        Compute channel correlation (Gram) matrix of a feature map.
        """
        b, c, h, w = feat.shape
        f = feat.view(b, c, h * w)
        g = torch.bmm(f, f.transpose(1, 2))
        return g / (c * h * w)

    def loss_fn(self, style, content, stylized):
        """
        Compute content (MSE) and style (Gram-MSE) losses.
        """
        f_s = self.vgg_extractor(style)
        f_c = self.vgg_extractor(content)
        f_t = self.vgg_extractor(stylized)
        c_loss = F.mse_loss(f_t[self.content_layer], f_c[self.content_layer])
        s_loss = 0.0
        for l in self.style_layers:
            g_s = self.gram_matrix(f_s[l])
            g_t = self.gram_matrix(f_t[l])
            s_loss += F.mse_loss(g_t, g_s)
        return self.content_loss_weight * c_loss + self.style_loss_weight * s_loss

    def training_step(self, batch, _):
        """Compute/log training loss."""
        style, content = batch["style"], batch["content"]
        style, content, stylized = self(style, content)
        loss = self.loss_fn(style, content, stylized)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        """Compute/log validation loss."""
        style, content = batch["style"], batch["content"]
        style, content, stylized = self(style, content)
        loss = self.loss_fn(style, content, stylized)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        """Collect stylized outputs."""
        style, content = batch["style"], batch["content"]
        style, content, stylized = self(style, content)
        self._test_outputs.append(stylized)

    def on_test_epoch_end(self):
        """Concatenate test outputs into test_results."""
        self.test_results = torch.cat(self._test_outputs, dim=0)
        self._test_outputs.clear()

    def configure_optimizers(self):
        """Setup optimizer and LR scheduler."""
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=self.lr_decay,
            patience=self.lr_patience
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}
        }
