'''segmenter.py

Defines the C-MARS segmentation model, combining a CLIP-based backbone with
an MBA neck, Transformer decoder, and final projector for referring image segmentation.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MBANeck_Mask2FormerStyle, Projector, TransformerDecoder, MultiModalBlock
from .clip import CLIP

class C_MARS(nn.Module):
    """
    C-MARS: Referring Image Segmentation model.
    Integrates a CLIP backbone, Mask2Former-style MBA neck, Transformer decoder,
    and dynamic convolution projector to generate binary segmentation masks.
    """

    def __init__(self, cfg):
        super().__init__()
        # Configure input dimensions
        input_shape = cfg.INPUT_SHAPE      # e.g., (3, 416, 416)
        txt_length = cfg.word_len          # maximum text token length

        # Initialize CLIP backbone for visual and textual embeddings
        self.backbone = CLIP(cfg, input_shape, txt_length)

        # Determine neck configuration based on backbone mode
        if hasattr(self.backbone, "mode") and self.backbone.mode == "vit":
            fake_down = True
            in_channels = [768, 768, 768]   # number of channels in selected ViT layers
        elif hasattr(self.backbone, "mode") and self.backbone.mode in ("convnext", "resnet"):
            fake_down = False
            in_channels = cfg.fpn_in        # e.g., [128, 256, 512]
        else:
            raise ValueError("Unknown backbone mode: {}".format(self.backbone.mode))

        # Multi-scale fusion neck: fuses visual features with text embeddings
        self.neck = MBANeck_Mask2FormerStyle(
            in_channels=in_channels,
            embed_dim=cfg.vis_dim,          # dimensionality of fused visual features
            text_dim=cfg.word_dim,          # dimensionality of text embeddings
            scales=(1, 3, 5),               # multi-scale kernel sizes for MBA
            mba_block_cls=MultiModalBlock,
            fake_downsample=fake_down
        )

        # Project text embedding to match visual feature dimension
        self.word_proj = nn.Linear(cfg.word_dim, cfg.vis_dim)

        # Transformer decoder for refining fused features with cross-modal attention
        self.decoder = TransformerDecoder(
            num_layers=cfg.num_layers,
            d_model=cfg.vis_dim,
            nhead=cfg.num_head,
            dim_ffn=cfg.dim_ffn,
            dropout=cfg.dropout,
            return_intermediate=cfg.intermediate
        )

        # Final dynamic convolution projector for mask prediction
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, kernel_size=3)

    def forward(self, img, word, mask=None):
        """
        Execute forward pass.

        Args:
            img (Tensor): input image batch of shape (B, 3, H, W)
            word (Tensor): tokenized text batch of shape (B, L)
            mask (Tensor, optional): ground truth masks for training of shape (B, 1, H, W)

        Returns:
            If training:
                Tuple(pred_logits, processed_mask, loss)
            Else:
                pred_logits: predicted mask logits of shape (B, 1, H, W)
        """
        # Prepare padding mask for text tokens
        pad_mask = (word == 0)

        # Extract multi-scale visual features and text embeddings
        vis_feats = self.backbone.extract_features(img)
        word_embed, text_state = self.backbone.encode_text(word)

        # Select feature maps for neck input
        if hasattr(self.backbone, "mode") and self.backbone.mode == "vit":
            features = [vis_feats['vit_l1'], vis_feats['vit_l6'], vis_feats['vit_l11']]
        else:
            features = [vis_feats['res2'], vis_feats['res3'], vis_feats['res4']]

        # Fuse visual and textual features
        fused, word_embed = self.neck(features, word_embed)
        if fused is None:
            raise ValueError("MBA neck returned None; expected a tensor.")

        # Project text embeddings to visual embedding dimension
        word_embed = self.word_proj(word_embed)

        # Refine fused features via Transformer decoder
        B, C, H, W = fused.size()
        decoded = self.decoder(fused, word_embed, pad_mask)
        decoded = decoded.view(B, C, H, W)

        # Generate mask logits and upsample to match input resolution
        pred = self.proj(decoded, text_state)
        if pred.shape[-2:] != img.shape[-2:]:
            pred = F.interpolate(pred, size=img.shape[-2:], mode='bilinear', align_corners=False)

        # Compute loss if ground truth mask is provided
        if self.training:
            if mask is not None and pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, size=pred.shape[-2:], mode='nearest').detach()
            loss = F.binary_cross_entropy_with_logits(pred, mask) if mask is not None else None
            return pred.detach(), mask, loss

        return pred.detach()
