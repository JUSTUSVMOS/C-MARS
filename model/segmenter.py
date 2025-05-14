'''segmenter.py

Defines the C-MARS segmentation model, combining a CLIP-based backbone with
an MBA neck, Transformer decoder, and final projector.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MBANeck, Projector, TransformerDecoder
from .clip import CLIP


class C_MARS(nn.Module):
    """
    Referring Image Segmentation (C-MARS) model using CLIP, MBA neck, decoder, and projector.
    """

    def __init__(self, cfg):
        super().__init__()
        # Input configuration
        input_shape = cfg.INPUT_SHAPE     # from TRAIN.INPUT_SHAPE
        txt_length = cfg.word_len         # max token length

        # CLIP backbone for image and text
        self.backbone = CLIP(cfg, input_shape, txt_length)

        # MBA neck: fuses visual features and text embeddings
        self.neck = MBANeck(
            in_channels=cfg.fpn_in,        # e.g., [128, 256, 512]
            embed_dim=cfg.vis_dim,
            text_dim=cfg.word_dim,         # e.g., 512
            # n_heads=cfg.num_head           # number of attention heads
        )

        # Transformer decoder for refined features
        self.decoder = TransformerDecoder(
            num_layers=cfg.num_layers,
            d_model=cfg.vis_dim,
            nhead=cfg.num_head,
            dim_ffn=cfg.dim_ffn,
            dropout=cfg.dropout,
            return_intermediate=cfg.intermediate
        )

        # Final projector: maps features to logits
        # Original signature: Projector(in_features, hidden_features, num_layers)
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)

    def forward(self, img, word, mask=None):
        """
        Forward pass.

        Args:
            img: Tensor[B, 3, H, W]
            word: Tensor[B, T]
            mask: Tensor[B, 1, H, W], optional for training

        Returns:
            (pred, mask_resized, loss) in training, else pred
        """
        # Create padding mask for text (True where token == 0)
        pad_mask = (word == 0)

        # Extract multi-scale visual features
        vis_feats = self.backbone.extract_features(img)
        # Encode text tokens
        word_embed, text_state = self.backbone.encode_text(word)

        # Prepare FPN inputs
        features = [vis_feats['res2'], vis_feats['res3'], vis_feats['res4']]

        # Fuse via MBA neck
        fused, word_embed = self.neck(features, word_embed)
        if fused is None:
            raise ValueError("MBA neck returned None; expected Tensor")

        # Decode features
        B, C, H, W = fused.size()
        decoded = self.decoder(fused, word_embed, pad_mask)
        decoded = decoded.view(B, C, H, W)

        # Generate mask logits
        pred = self.proj(decoded, text_state)

        if self.training:
            # Resize ground-truth mask to match pred
            if mask is not None and pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()
            # Compute BCE loss
            loss = F.binary_cross_entropy_with_logits(pred, mask)
            return pred.detach(), mask, loss

        return pred.detach()
