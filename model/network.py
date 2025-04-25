import torch
import numpy as np
from os.path import join
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
        extraction_layers=[8, 15, 20, 26, 31, 35],
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
        betas=(0.9, 0.999),
        results_path="predictions"
    ):
        super().__init__()
        self.save_hyperparameters()
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
        self.vgg_extractor = VGGFeatureExtractor(extraction_layers=[8, 15, 20, 26, 31, 35])
        self.vgg_layers = [f"state_{i}" for i in range(len(extraction_layers))]

        # Training params
        self.img_height = img_height
        self.img_width = img_width
        self.lambdas = lambdas
        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_decay = lr_decay
        self.betas = betas

        # Test outputs
        self.results_path = results_path
        self._test_outputs = {
            "style": [],
            "content": [],
            "stylized": [],
            "reverse_stylized": []
        }
        self.test_results = None

    def forward(self, style, content):
        """
        Stylize content using style.
        """
        """
        # Patching (add checks for the identity losses)
        if style.size(-1) == 256:
            s_patches = self.content_patcher(style)
        elif style.size(-1) == 512:
            s_patches = self.style_patcher(style)
            
        if content.size(-1) == 256:
            c_patches = self.content_patcher(content)
        elif content.size(-1) == 512:
            c_patches = self.style_patcher(content)
        """
        s_patches = self.style_patcher(style)
        c_patches = self.content_patcher(content)
        pos_embed = self.cape(c_patches)
        c_patches = c_patches + pos_embed
        
        c_patches = c_patches.flatten(2).permute(0, 2, 1)
        s_patches = s_patches.flatten(2).permute(0, 2, 1)

        # Encoder
        c_enc = self.content_encoder(c_patches) + pos_embed.flatten(2).permute(0, 2, 1)  # CAPE on content embeddings
        s_enc = self.style_encoder(s_patches)
        del c_patches, s_patches  # clear up space

        # Decoder
        tokens = self.decoder(target_tokens=c_enc, memory=s_enc)
        del c_enc, s_enc  # clear up space
        
        stylized = self.cnn_decoder(tokens)  # reshaping happens internally
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
        b, c, h, w = feat.size()
        feat_flat = feat.view(b, c, -1)
        var = feat_flat.var(dim=2) + eps
        std = var.sqrt().view(b, c, 1, 1)
        mean = feat_flat.mean(dim=2).view(b, c, 1, 1)
        return mean, std
        
    def normalize(self, feat):
        mean, std = self.calc_stats(feat)
        return (feat - mean) / std
        
    def loss_fn(self, style, content, stylized, identity_style, identity_content, idx):
        """
        Compute content, style, and identity losses without resizing to 224×224.
        """
        # 1) VGG features
        f_s = self.vgg_extractor(style)
        f_c = self.vgg_extractor(content)
        f_t = self.vgg_extractor(stylized)
    
        # 2) Content loss on the two deepest layers
        c_loss = (
            F.mse_loss(
                self.normalize(f_t[self.vgg_layers[-1]]),
                self.normalize(f_c[self.vgg_layers[-1]])
            )
            + F.mse_loss(
                self.normalize(f_t[self.vgg_layers[-2]]),
                self.normalize(f_c[self.vgg_layers[-2]])
            )
        )
    
        # 3) Style loss across all extracted layers
        s_loss = 0.0
        for l in self.vgg_layers:
            mean_s, std_s = self.calc_stats(f_s[l])
            mean_t, std_t = self.calc_stats(f_t[l])
            s_loss += F.mse_loss(mean_t, mean_s) + F.mse_loss(std_t, std_s)
    
        # 4) Identity loss 1 (pixel‐level)
        i_loss1 = F.mse_loss(identity_style, style) + F.mse_loss(identity_content, content)
    
        # 5) Identity loss 2 (feature‐level)
        i_s = self.vgg_extractor(identity_style)
        i_c = self.vgg_extractor(identity_content)
        i_loss2 = 0.0
        for l in self.vgg_layers:
            i_loss2 += F.mse_loss(i_s[l], f_s[l]) + F.mse_loss(i_c[l], f_c[l])
    
        # (optional) logging  
        if idx % 100 == 0:
            print(f"[{idx}] c:{c_loss.item():.4f}  s:{s_loss.item():.4f}  i1:{i_loss1.item():.4f}  i2:{i_loss2.item():.4f}")
    
        # 6) Weighted sum
        loss_main = (
            self.lambdas[0] * c_loss
            + self.lambdas[1] * s_loss
            + self.lambdas[2] * i_loss1
            + self.lambdas[3] * i_loss2
        )
        return loss_main
        
    def on_after_backward(self):
        """Compute the total L2 norm of all gradients."""
        total_norm_sq = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm_sq ** 0.5

        if self.global_step % 100 == 0:
            print(f"[step {self.global_step}] grad_norm: {total_norm:.4f}")

    def training_step(self, batch, idx):
        """Compute/log training loss."""
        style, content = batch["style"], batch["content"]
        stylized = self(style, content)
        identity_style = self(style, style)
        identity_content = self(content, content)
        
        loss = self.loss_fn(style, content, stylized, identity_style, identity_content, idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, idx):
        """Compute/log validation loss."""
        style, content = batch["style"], batch["content"]
        stylized = self(style, content)
        identity_style = self(style, style)
        identity_content = self(content, content)
        
        loss = self.loss_fn(style, content, stylized, identity_style, identity_content, idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        """Collect stylized outputs."""
        style, content = batch["style"] * 255.0, batch["content"] * 255.0
        stylized = self(style, content) * 255.0
        reverse_stylized = self(content, style) * 255.0
        
        # Store test results
        self._test_outputs["style"].append((style * 255.0).detach().cpu())
        self._test_outputs["content"].append((content * 255.0).detach().cpu())
        self._test_outputs["stylized"].append(stylized.detach().cpu())
        self._test_outputs["reverse_stylized"].append(reverse_stylized.detach().cpu())

    def on_test_epoch_end(self):
        """Concatenate test outputs into test_results."""
        for k in self._test_outputs.keys():
            result = torch.cat(self._test_outputs[k], dim=0).numpy()
            results_filepath = join(self.results_path, f"{k}.npy")
            np.save(results_filepath, result)
            self._test_outputs[k].clear()

    def configure_optimizers(self):
        """Setup optimizer and LR scheduler."""
        """
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
        """
        # Implementation below is from GitHub implementation
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=self.betas)

        def lr_fn(step):
            # warm‑up phase
            if step < 1e4:
                return 0.1 * (1.0 + 3e-4 * step)
            # decay phase
            else:
                lr = 2e-4 / (1.0 + self.lr_decay * (step - 1e4))
                return lr / self.lr

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(opt, lr_fn),
            'interval': 'step',
            'frequency': 1,
        }
        return {'optimizer': opt, 'lr_scheduler': scheduler}
