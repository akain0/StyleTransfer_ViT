import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from model.patching import ContentPatching, StylePatching
from model.cape import CAPE
from model.encoders import ContentTransformerEncoder, StyleTransformerEncoder
from model.decoders import TransformerDecoder, CNNDecoder
from model.vgg import VGGFeatureExtractor

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
        # VGG
        img_height=224,
        img_width=224,
        # Patching
        n=18,
        s=1,
        # Content Encoder
        patch_size_content=8,
        n_head_content=8,
        dim_feedforward_content=2048,
        dropout_content=0.1,
        n_layers_content=6,
        # Style Encoder
        patch_size_style=8,
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
        lambdas=[10, 7, 50, 1],
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
        self.cnn_decoder = CNNDecoder(embed_dim=d_model)

        # VGG
        self.vgg_extractor = VGGFeatureExtractor()
        self.vgg_layers = ["state_1", "state_2", "state_3", "state_4", "state_5", "state_6"]

        # Training params
        self.img_height = img_height
        self.img_width = img_width
        self.lambdas = lambdas
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_decay = lr_decay
        self.betas = betas

        # Test outputs
        self._test_outputs = {
            "style": [],
            "content": [],
            "stylized": []
        }
        self.test_results = None

    def forward(self, style, content):
        """
        Stylize content using style.
        """
        # Patching
        c_patches = self.content_patcher(content)
        s_patches = self.style_patcher(style)
        c_patches = self.cape(c_patches)

        # Encoder
        c_enc = self.content_encoder(c_patches)
        s_enc = self.style_encoder(s_patches)
        del c_patches, s_patches  # clear up space

        # Decoder
        tokens = self.decoder(target_tokens=c_enc, memory=s_enc)
        del c_enc, s_enc  # clear up space
        
        stylized = self.cnn_decoder(tokens)
        del tokens  # clear up space
        return stylized
    
    def gram_matrix(self, feat):
        """
        Compute channel correlation (Gram) matrix of a feature map.
        Not being used currently.
        """
        b, c, h, w = feat.shape
        f = feat.view(b, c, h * w)
        g = torch.bmm(f, f.transpose(1, 2))
        return g / (c * h * w)
        
    def calc_stats(self, feat, eps=1e-5):
        """
        Compute channel‐wise mean and stddev over spatial dims.
        """
        b, c = feat.size()[:2]
        feat_flat = feat.view(b, c, -1)
        var = feat_flat.var(dim=2) + eps
        std = var.sqrt().view(b, c, 1, 1)
        mean = feat_flat.mean(dim=2).view(b, c, 1, 1)
        return mean, std

    def loss_fn(self, style, content, stylized, identity_style, identity_content):
        """
        Use this in case the current implementation does not work:
        def loss_fn(self, style, content, stylized):
            # Compute content (MSE) and style (Gram-MSE) losses.
            f_s = self.vgg_extractor(style)
            f_c = self.vgg_extractor(content)
            f_t = self.vgg_extractor(stylized)
            c_loss = F.mse_loss(f_t[self.content_layer], f_c[self.content_layer])
            s_loss = 0.0
            for l in self.style_layers:
                g_s = self.gram_matrix(f_s[l])
                g_t = self.gram_matrix(f_t[l])
                s_loss += F.mse_loss(g_t, g_s)
            return (self.content_loss_weight * c_loss) + (self.style_loss_weight * s_loss)
        """
        device = style.device
        
        # Content loss
        with torch.no_grad():  # no grad because they are targets
            style_vgg = F.interpolate(
                style,
                (self.img_height, self.img_width),
                mode="bicubic",
                align_corners=False
            )  # standardize image dimensions for VGG
            content_vgg = F.interpolate(
                content,
                (self.img_height, self.img_width),
                mode="bicubic",
                align_corners=False
            )  # standardize image dimensions for VGG
            f_s = self.vgg_extractor(style_vgg)
            f_c = self.vgg_extractor(content_vgg)
            del style_vgg, content_vgg  # clear up space

        stylized_vgg = F.interpolate(
            stylized,
            (self.img_height, self.img_width),
            mode="bicubic",
            align_corners=False
        )  # standardize image dimensions for VGG
        
        f_t = self.vgg_extractor(stylized_vgg)
            
        # move to CPU to free GPU memory
        f_s = {l: v.detach().cpu() for l, v in f_s.items()}
        f_c = {l: v.detach().cpu() for l, v in f_c.items()}

        c_loss = F.mse_loss(f_t[self.vgg_layers[-1]], f_c[self.vgg_layers[-1]].to(device))
        
        # Style loss
        s_loss = 0.0
        for l in self.vgg_layers:
            mean_s, std_s = self.calc_stats(f_s[l].to(device))
            mean_t, std_t = self.calc_stats(f_t[l])
            s_loss += F.mse_loss(mean_t, mean_s)
            s_loss += F.mse_loss(std_t, std_s)
        s_loss /= len(self.vgg_layers)
        del stylized_vgg, f_t  # clear up space
        
        # Identity loss 1
        i_loss1 = F.mse_loss(identity_style, style) + F.mse_loss(identity_content, content)

        # Identity loss 2
        identity_style_vgg = F.interpolate(
            identity_style,
            (self.img_height, self.img_width),
            mode="bicubic",
            align_corners=False
        )  # standardize image dimensions for VGG
        identity_content_vgg = F.interpolate(
            identity_content,
            (self.img_height, self.img_width),
            mode="bicubic",
            align_corners=False
        )  # standardize image dimensions for VGG
        
        i_s = self.vgg_extractor(identity_style_vgg)
        i_c = self.vgg_extractor(identity_content_vgg)
        del identity_style_vgg, identity_content_vgg  # clear up space
        
        i_loss2 = 0.0
        for l in self.vgg_layers:
            i_loss2 += F.mse_loss(i_s[l], f_s[l].to(device)) + F.mse_loss(i_c[l], f_c[l].to(device))
        
        i_loss2 /= len(self.vgg_layers)
        del f_s, f_c, i_s, i_c  # clear up space
        
        # Total loss
        loss_main = (self.lambdas[0] * c_loss) + (self.lambdas[1] * s_loss) + \
                (self.lambdas[2] * i_loss1) + (self.lambdas[3] * i_loss2)
        return loss_main
        

    def training_step(self, batch, _):
        """Compute/log training loss."""
        style, content = batch["style"], batch["content"]
        stylized = self(style, content)
        identity_style = self(style, style)
        identity_content = self(content, content)
        
        loss = self.loss_fn(style, content, stylized, identity_style, identity_content)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        """Compute/log validation loss."""
        style, content = batch["style"], batch["content"]
        stylized = self(style, content)
        identity_style = self(style, style)
        identity_content = self(content, content)
        
        loss = self.loss_fn(style, content, stylized, identity_style, identity_content)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        """Collect stylized outputs."""
        style, content = batch["style"], batch["content"]
        stylized = self(style, content)
        
        # Store test results
        self._test_outputs["style"].append(style.detach().cpu())
        self._test_outputs["content"].append(content.detach().cpu())
        self._test_outputs["stylized"].append(stylized.detach().cpu())

    def on_test_epoch_end(self):
        """Concatenate test outputs into test_results."""
        self.test_results = {}
        for k in self._test_outputs.keys():
            self.test_results[k] = torch.cat(self._test_outputs[k], dim=0)
            self._test_outputs[k].clear()

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
