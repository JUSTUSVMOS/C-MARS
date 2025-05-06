import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import FPN, Projector, TransformerDecoder
from .clip import CLIP  

class CRIS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        input_shape = cfg.INPUT_SHAPE  # 修改这里，从TRAIN部分获取INPUT_SHAPE
        txt_length = cfg.word_len  
        self.backbone = CLIP(cfg, input_shape, txt_length)
        # Multi-Modal FPN
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        # Decoder
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)
        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)

    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # Extract visual features from ConvNeXt
        vis = self.backbone.extract_features(img)  # 从ConvNeXt中提取特征
        word, state = self.backbone.encode_text(word)
        
        
        # print("Visual features extracted:")
        # for key, value in vis.items():
        #     print(f"{key}: {value.shape}")

        # Process visual features through FPN
        imgs = [vis['res2'], vis['res3'], vis['res4']]
        # 调试输出：检查imgs是否是包含三个特征图的列表
        # print("Features passed to FPN:")
        # for i, feature in enumerate(imgs):
        #     print(f"Feature {i}: {feature.shape}")
            
        try:
            fq = self.neck(imgs, state)
            # print(f"FPN output shape: {fq.shape}")
        except Exception as e:
            # print(f"Error in FPN: {e}")
            raise

        if fq is None:
            raise ValueError("FPN did not return a valid output")

        b, c, h, w = fq.size()
        fq = self.decoder(fq, word, pad_mask)
        fq = fq.reshape(b, c, h, w)

        # Generate final predictions
        pred = self.proj(fq, state)

        if self.training:
            # Resize mask to match prediction size
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()
            loss = F.binary_cross_entropy_with_logits(pred, mask)
            return pred.detach(), mask, loss
        else:
            return pred.detach()
