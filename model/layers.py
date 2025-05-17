import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Common helper layers ----------

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


# ---------- CoordConv  (used in original CRIS) ----------

class CoordConv(nn.Module):
    """Conv2d + (x,y) coordinate channels."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = conv_layer(in_channels + 2, out_channels, kernel_size, padding, stride)

    def _add_coord(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.size()
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device)
        )
        coord = torch.stack([xs.expand(b, -1, -1), ys.expand(b, -1, -1)], dim=1)  # (B,2,H,W)
        return torch.cat([x, coord], dim=1)

    def forward(self, x):
        return self.conv(self._add_coord(x))


# ---------- Projector (dynamic per‑sample conv) ----------

class Projector(nn.Module):
    """Same as CRIS: transforms fused feature + text state into mask logits."""
    def __init__(self, word_dim=1024, in_dim=1024, kernel_size=3):
        super().__init__()
        self.k = kernel_size
        out_dim = in_dim * kernel_size * kernel_size + 1  # +bias
        self.conv_vis = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1)
        )
        self.fc_txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word_state):
        # x: (B,C,H,W)  word_state: (B,word_dim)
        x = self.conv_vis(x)
        B, C, H, W = x.shape
        x = x.reshape(1, B * C, H, W)                    # grouped conv trick
        params = self.fc_txt(word_state)
        weight, bias = params[:, :-1], params[:, -1]
        weight = weight.reshape(B, C, self.k, self.k)
        out = F.conv2d(x, weight, bias=bias, padding=self.k // 2, groups=B)
        return out.transpose(0, 1)                        # (B,1,H,W)


# ---------- Transformer Decoder (unchanged CRIS version) ----------

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_ffn, dropout, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_ffn, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    def _pos1d(self, d_model, length, device):
        pe = torch.zeros(length, d_model, device=device)
        pos = torch.arange(length, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, device=device) * -(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(1)  # (L,1,D)

    def _pos2d(self, d_model, h, w, device):
        if d_model % 4 != 0:
            raise ValueError("d_model must be multiple of 4")
        pe = torch.zeros(d_model, h, w, device=device)
        div = torch.exp(torch.arange(0, d_model//2, 2, device=device) * -(math.log(10000.0)/(d_model//2)))
        grid_w = torch.arange(w, device=device).unsqueeze(1)
        grid_h = torch.arange(h, device=device).unsqueeze(1)
        pe[0::4, :, :] = torch.sin(grid_w * div).transpose(0,1).unsqueeze(1).repeat(1,h,1)
        pe[1::4, :, :] = torch.cos(grid_w * div).transpose(0,1).unsqueeze(1).repeat(1,h,1)
        pe[2::4, :, :] = torch.sin(grid_h * div).transpose(0,1).unsqueeze(2).repeat(1,1,w)
        pe[3::4, :, :] = torch.cos(grid_h * div).transpose(0,1).unsqueeze(2).repeat(1,1,w)
        return pe.view(d_model, -1).T.unsqueeze(1)        # (HW,1,D)

    def forward(self, vis, txt, pad_mask):
        B, C, H, W = vis.shape
        L = txt.size(1)
        pos_v = self._pos2d(C, H, W, vis.device)
        pos_t = self._pos1d(C, L, txt.device)
        v_seq = vis.flatten(2).permute(2,0,1)  # (HW,B,C)
        t_seq = txt.permute(1,0,2)            # (L ,B,C)
        out = v_seq
        inter = []
        for layer in self.layers:
            out = layer(out, t_seq, pos_v, pos_t, pad_mask)
            if self.return_intermediate:
                inter.append(self.norm(out).permute(1,2,0).view(B,C,H,W))
        out = self.norm(out).permute(1,2,0).view(B,C,H,W)
        if self.return_intermediate:
            inter[-1] = out
            return inter
        return out

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _add_pos(x, p):
        return x if p is None else x + p.to(x.device)

    def forward(self, v, t, pos_v, pos_t, pad_mask):
        q = k = self._add_pos(self.norm1(v), pos_v)
        v = v + self.drop(self.self_attn(q,k,v)[0])
        q = self._add_pos(self.norm2(v), pos_v)
        k = self._add_pos(t, pos_t)
        v = v + self.drop(self.cross_attn(q,k,t, key_padding_mask=pad_mask)[0])
        v = v + self.drop(self.ffn(self.norm3(v)))
        return v

# ---------- Bidirectional Cross‑Attention Block with MBA Text->Image ----------
class MultiModalBlock(nn.Module):
    """Multi-scale Text→Image cross-attention only, with softmax-normalized λ."""
    def __init__(self, v_dim, t_dim, scales=(1, 3, 5)):
        super().__init__()
        self.scales = scales
        # Q is projected from visual features, K/V from text tokens
        self.q_lin  = nn.Linear(v_dim, v_dim)
        self.k_txt  = nn.Linear(t_dim, v_dim)
        self.v_txt  = nn.Linear(t_dim, v_dim)
        # Initialize λ parameters to zeros so that softmax yields uniform weights initially
        self.lambda_r = nn.Parameter(torch.zeros(len(scales)))
        # Scaling factor for attention scores
        self.scale    = v_dim ** -0.5

    def _window_pool(self, v: torch.Tensor, k: int) -> torch.Tensor:
        """
        Apply average pooling with kernel size k to each channel, 
        padded so output stays the same spatial dimensions.
        If k=1, return the feature unchanged.
        """
        if k == 1:
            return v
        pad = k // 2
        return F.avg_pool2d(v, kernel_size=k, stride=1, padding=pad)

    def forward(self, v_feat: torch.Tensor, t_tok: torch.Tensor):
        """
        Args:
            v_feat: Visual feature map of shape (B, C, H, W)
            t_tok : Text token embeddings of shape (B, L, Ct)

        Returns:
            Updated visual features and unchanged text tokens:
            - v_feat': (B, C, H, W)
            - t_tok : (B, L, Ct)
        """
        B, C, H, W = v_feat.shape
        N = H * W

        # Project text tokens to get keys and values
        k_txt = self.k_txt(t_tok)  # (B, L, C)
        v_txt = self.v_txt(t_tok)  # (B, L, C)

        # Normalize λ across scales so they sum to 1
        weights = F.softmax(self.lambda_r, dim=0)  # (num_scales,)

        # Accumulate multi-scale attention outputs
        agg = 0
        for idx, k in enumerate(self.scales):
            # Pool visual feature in a k×k window (or leave as is if k=1)
            v_pool = self._window_pool(v_feat, k)           # (B, C, H, W)
            v_flat = v_pool.view(B, C, N).permute(0, 2, 1)   # (B, N, C)
            q      = self.q_lin(v_flat)                     # (B, N, C)

            # Compute attention logits and distribution
            attn_logits = torch.einsum('bnc,blc->bnl', q, k_txt) * self.scale  # (B, N, L)
            attn        = attn_logits.softmax(dim=-1)                        # (B, N, L)

            # Weight text values by attention and sum
            attn_out    = torch.einsum('bnl,blc->bnc', attn, v_txt)           # (B, N, C)

            # Weighted by normalized λ for this scale
            agg = agg + weights[idx] * attn_out

        # Reshape back to feature map and add residual to v_feat
        out_vis = agg.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)
        return v_feat + out_vis, t_tok


# ---------- MBA Neck (concat version) ----------

class MBANeck(nn.Module):
    """Multi‑scale Bidirectional Attention neck (concat + dynamic upsample)."""
    def __init__(self, in_channels=[1024, 1024, 1024], embed_dim=1024, text_dim=1024):
        super().__init__()
        self.v_proj = nn.ModuleList([nn.Conv2d(c, embed_dim, 1) for c in in_channels[::-1]])
        self.t_proj = nn.Linear(text_dim, embed_dim)
        self.blocks = nn.ModuleList([
            MultiModalBlock(embed_dim, embed_dim) for _ in range(3)
        ])
        self.merge_conv = nn.Sequential(
            nn.Conv2d(embed_dim*3, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

    def forward(self, feats, txt_tok):
        txt_tok = self.t_proj(txt_tok)
        up_feats = []
        target_size = feats[0].shape[-2:]
        for v, proj, blk in zip(feats[::-1], self.v_proj, self.blocks):
            v = proj(v)
            v, txt_tok = blk(v, txt_tok)
            v_up = F.interpolate(v, size=target_size, mode='bilinear', align_corners=False)
            up_feats.append(v_up)
        fused = torch.cat(up_feats, dim=1)
        fused = self.merge_conv(fused)
        fused = F.avg_pool2d(fused, kernel_size=2, stride=2)
        return fused, txt_tok
