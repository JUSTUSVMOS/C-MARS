import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.GELU()
    )


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias),
        nn.BatchNorm1d(out_dim),
        nn.GELU()
    )


class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        # Convolution that adds coordinate channels before applying kernel
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        # Generate normalized coordinate grids in range [-1, 1]
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        # Concatenate coordinate channels to input
        return torch.cat([input, coord_feat], 1)

    def forward(self, x):
        x = self.add_coord(x)
        return self.conv1(x)


class Projector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # Upsample visual features from stride-16 to stride-4
        self.vis = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1)
        )
        # Predict per-sample convolution kernels + bias from text embeddings
        out_dim = in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        '''
            x: visual features (B, C, H, W)
            word: text embeddings (B, word_dim)
        '''
        x = self.vis(x)
        B, C, H, W = x.size()
        # Reshape for grouped convolution: (1, B*C, H, W)
        x = x.reshape(1, B * C, H, W)
        # Generate weights and bias per sample
        params = self.txt(word)
        weight, bias = params[:, :-1], params[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Apply grouped convolution across batch dimension
        out = F.conv2d(
            x,
            weight,
            padding=self.kernel_size // 2,
            groups=B,
            bias=bias
        )
        # Reshape back to (B, 1, H, W)
        return out.transpose(0, 1)


class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ffn,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        1D sinusoidal positional encoding.
        """
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        pe = torch.zeros(length, d_model)
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe.unsqueeze(1)  # (length, 1, d_model)

    @staticmethod
    def pos2d(d_model, height, width):
        """
        2D sinusoidal positional encoding.
        """
        if d_model % 4 != 0:
            raise ValueError(f"d_model must be multiple of 4, got {d_model}")
        pe = torch.zeros(d_model, height, width)
        half = d_model // 2
        div_term = torch.exp(torch.arange(0, half, 2) * -(math.log(10000.0) / half))
        # Width encoding
        pos_w = torch.arange(width).unsqueeze(1)
        pe[0:half:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:half:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        # Height encoding
        pos_h = torch.arange(height).unsqueeze(1)
        pe[half::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[half+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        # Flatten to (H*W, 1, d_model)
        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)

    def forward(self, vis, txt, pad_mask):
        '''
            vis: visual features (B, C, H, W)
            txt: text tokens     (B, L, d_model)
            pad_mask: padding mask for text (B, L)
        '''
        B, C, H, W = vis.size()
        _, L, D = txt.size()
        # Generate positional encodings
        vis_pos = self.pos2d(C, H, W)
        txt_pos = self.pos1d(D, L)
        # Flatten and permute to sequence format
        vis_seq = vis.reshape(B, C, -1).permute(2, 0, 1)
        txt_seq = txt.permute(1, 0, 2)
        output = vis_seq
        intermediate = []
        # Pass through decoder layers
        for layer in self.layers:
            output = layer(output, txt_seq, vis_pos, txt_pos, pad_mask)
            if self.return_intermediate:
                intermediate.append(self.norm(output).permute(1, 2, 0))
        # Final normalization
        output = self.norm(output).permute(1, 2, 0)
        if self.return_intermediate:
            intermediate[-1] = output
            return intermediate
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=9,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # Normalization for attention blocks
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # Self- and cross-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout,
            kdim=d_model, vdim=d_model
        )
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, d_model)
        )
        # Additional norms and dropouts
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask):
        '''
            vis:    visual sequence (HW, B, d_model)
            txt:    text sequence   (L,  B, d_model)
            vis_pos: (HW, 1, d_model)
            txt_pos: (L,   1, d_model)
            pad_mask: (B, L)
        '''
        # Self-attention on visual tokens
        v2 = self.norm1(vis)
        q = k = self.with_pos_embed(v2, vis_pos)
        v2 = self.self_attn(q, k, value=v2)[0]
        vis = vis + self.dropout1(self.self_attn_norm(v2))
        # Cross-attention from visual to text
        v2 = self.norm2(vis)
        v2 = self.multihead_attn(
            query=self.with_pos_embed(v2, vis_pos),
            key=self.with_pos_embed(txt, txt_pos),
            value=txt,
            key_padding_mask=pad_mask
        )[0]
        vis = vis + self.dropout2(self.cross_attn_norm(v2))
        # Feed-forward
        v2 = self.norm3(vis)
        v2 = self.ffn(v2)
        return vis + self.dropout3(v2)


class FPN(nn.Module):
    def __init__(self,
                 in_channels=[128, 256, 512],  # Backbone output channels
                 out_channels=[256, 512, 1024], # FPN intermediate channels
                 state_dim=640):                # State embedding dimension
        super().__init__()
        # Project state vector to match deepest feature channels
        self.state_proj = nn.Linear(state_dim, in_channels[2])
        self.txt_proj = nn.Sequential(
            nn.Linear(in_channels[2], out_channels[2]),
            nn.GELU()
        )
        # Fusion level 5: combine v5 and state
        self.f1_v_proj = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1)
        self.norm_layer = nn.Sequential(
            nn.BatchNorm2d(out_channels[2]),
            nn.GELU()
        )
        # Fusion level 4: combine v4 and intermediate f5
        self.f2_v_proj = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=3, padding=1)
        self.f2_cat    = nn.Conv2d(out_channels[2]+out_channels[1], out_channels[1], kernel_size=1)
        # Fusion level 3: combine v3 and intermediate f4
        self.f3_v_proj = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=3, padding=1)
        self.f3_cat    = nn.Conv2d(out_channels[0]+out_channels[1], out_channels[0], kernel_size=1)
        # Aggregate all levels
        self.aggr = nn.Conv2d(
            out_channels[0]+out_channels[1]+out_channels[2],
            out_channels[1],
            kernel_size=1
        )
        # Final refinement with coordconv
        self.coordconv = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels[1]),
            nn.GELU(),
            nn.Conv2d(out_channels[1], out_channels[1], kernel_size=3, padding=1)
        )

    def forward(self, imgs, state):
        '''
            imgs:  list of feature maps [v3, v4, v5]
            state: state embedding tensor (B, state_dim)
        '''
        v3, v4, v5 = imgs
        # Project state and reshape for broadcasting
        s = self.txt_proj(self.state_proj(state)).unsqueeze(-1).unsqueeze(-1)
        # Fusion level 5
        f5 = self.norm_layer(self.f1_v_proj(v5) * s)
        f5 = F.interpolate(f5, size=(52, 52), mode='bilinear', align_corners=False)
        # Fusion level 4
        f4 = F.interpolate(self.f2_v_proj(v4), size=(52, 52), mode='bilinear', align_corners=False)
        f4 = self.f2_cat(torch.cat([f5, f4], dim=1))
        # Fusion level 3
        f3 = F.interpolate(self.f3_v_proj(v3), size=(52, 52), mode='bilinear', align_corners=False)
        f3 = self.f3_cat(torch.cat([f4, f3], dim=1))
        # Aggregate and refine
        fq = self.aggr(torch.cat([f3, f4, f5], dim=1))
        return self.coordconv(fq)


# ---------- MBA Neck (Bi-directional Cross-Attention) ----------
class MultiModalBlock(nn.Module):
    """
    Single block of bidirectional cross-attention between
    image features and text tokens.
    """
    def __init__(self, v_dim, t_dim, n_heads=8):
        super().__init__()
        self.v2t_attn = nn.MultiheadAttention(embed_dim=t_dim, num_heads=n_heads)
        self.t2v_attn = nn.MultiheadAttention(embed_dim=v_dim, num_heads=n_heads)
        self.v_ln = nn.LayerNorm(v_dim)
        self.t_ln = nn.LayerNorm(t_dim)

    def forward(self, v_feat, t_tok):
        '''
            v_feat: image feature map (B, C, H, W)
            t_tok: text token embeddings (B, L, C_t)
        '''
        B, C, H, W = v_feat.shape
        # Image→Text cross-attention
        t_in = self.t_ln(t_tok)
        v_flat = v_feat.flatten(2).permute(2, 0, 1)
        t_q = t_in.permute(1, 0, 2)
        t_ctx, _ = self.v2t_attn(t_q, v_flat, v_flat)
        t_tok = t_tok + t_ctx.permute(1, 0, 2)
        # Text→Image cross-attention
        v_in = self.v_ln(v_flat)
        t_kv = t_tok.permute(1, 0, 2)
        v_ctx, _ = self.t2v_attn(v_in, t_kv, t_kv)
        v_feat = v_feat + v_ctx.permute(1, 2, 0).view(B, C, H, W)
        return v_feat, t_tok


class MBANeck(nn.Module):
    """
    Multi-scale bidirectional attention neck, replacing FPN.
    Args:
        in_channels: list of channels from backbone stages [C3, C4, C5]
        embed_dim: embedding dimension for both visual and text tokens
    """
    def __init__(self,
                 in_channels=[128, 256, 512],
                 embed_dim=512,
                 n_heads=8):
        super().__init__()
        # Project backbone outputs to embed_dim
        self.v_proj = nn.ModuleList([
            nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels[::-1]
        ])
        self.t_proj = nn.Linear(embed_dim, embed_dim)
        # Bidirectional cross-attention blocks
        self.mba = nn.ModuleList([
            MultiModalBlock(embed_dim, embed_dim, n_heads) for _ in range(3)
        ])
        # Final refinement
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

    def forward(self, imgs, txt_tok):
        '''
            imgs: list of feature maps [v3, v4, v5]
            txt_tok: text token embeddings (B, L, embed_dim)
        '''
        v_feats = imgs[::-1]  # process v5, v4, v3 in order
        t_feat = self.t_proj(txt_tok)
        agg = None
        for v, proj, block in zip(v_feats, self.v_proj, self.mba):
            v = proj(v)
            v, t_feat = block(v, t_feat)
            v_up = F.interpolate(v, size=(52, 52), mode='bilinear', align_corners=False)
            agg = v_up if agg is None else agg + v_up
        return self.out_conv(agg)