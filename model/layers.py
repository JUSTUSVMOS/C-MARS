import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.GELU())


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.GeLU())


class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class Projector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        x = self.vis(x)
        B, C, H, W = x.size()
        # 1, b*256, 104, 104
        x = x.reshape(1, B * C, H, W)
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        out = F.conv2d(x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
        out = out.transpose(0, 1)
        # b, 1, 104, 104
        return out


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
            TransformerDecoderLayer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_ffn,
                                    dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis, txt, pad_mask):
        '''
            vis: b, 512, h, w
            txt: b, L, 512
            pad_mask: b, L
        '''
        B, C, H, W = vis.size()
        _, L, D = txt.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        txt_pos = self.pos1d(D, L)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        txt = txt.permute(1, 0, 2)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, txt, vis_pos, txt_pos, pad_mask)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=9,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.GELU(), nn.Dropout(dropout),
                                 nn.LayerNorm(dim_feedforward),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
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
            vis: 26*26, b, 512
            txt: L, b, 512
            vis_pos: 26*26, 1, 512
            txt_pos: L, 1, 512
            pad_mask: b, L
        '''
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2 = self.multihead_attn(query=self.with_pos_embed(vis2, vis_pos),
                                   key=self.with_pos_embed(txt, txt_pos),
                                   value=txt,
                                   key_padding_mask=pad_mask)[0]
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.dropout2(vis2)
        # FFN
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout3(vis2)
        return vis


class FPN(nn.Module):
    def __init__(self,
                 in_channels=[128, 256, 512],  # ConvNeXt的輸出通道數
                 out_channels=[256, 512, 1024],  # 輸出的特徵圖通道數
                 state_dim=640):  # state張量的維度
        super(FPN, self).__init__()
        self.state_proj = nn.Linear(state_dim, in_channels[2])  # 640投射到512
        self.txt_proj = nn.Sequential(
            nn.Linear(in_channels[2], out_channels[2]),  # 512投射到1024
            nn.GELU()
        )
        # 融合 1：v5 & seq -> f_5: b, 1024, 14, 14
        self.f1_v_proj = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, padding=0)
        self.norm_layer = nn.Sequential(
            nn.BatchNorm2d(out_channels[2]),
            nn.GELU()
        )
        # 融合 2：v4 & fm -> f_4: b, 512, 28, 28
        self.f2_v_proj = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=3, padding=1)
        self.f2_cat = nn.Conv2d(out_channels[2] + out_channels[1], out_channels[1], kernel_size=1, padding=0)
        # 融合 3：v3 & fm_mid -> f_3: b, 256, 28, 28
        self.f3_v_proj = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=3, padding=1)
        self.f3_cat = nn.Conv2d(out_channels[0] + out_channels[1], out_channels[0], kernel_size=1, padding=0)  
        # 聚合
        self.aggr = nn.Conv2d(out_channels[0] + out_channels[1] + out_channels[2], out_channels[1], kernel_size=1, padding=0)
        self.coordconv = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels[1]),
            nn.GELU(),
            nn.Conv2d(out_channels[1], out_channels[1], kernel_size=3, padding=1)
        )

    def forward(self, imgs, state):
        v3, v4, v5 = imgs  # 假設 imgs 是一個包含三個特徵圖的列表
        state = self.state_proj(state)
        # print(f"State shape after projection in FPN: {state.shape}")
        # 融合 1：b, 1024, 14, 14
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(-1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)  # 確保 f5 的輸入通道數是 512
        f5 = self.norm_layer(f5 * state)
        f5 = F.interpolate(f5, size=(52, 52), mode='bilinear', align_corners=False)  # 上采樣到 52x52
        
        # 融合 2：b, 512, 28, 28
        f4 = self.f2_v_proj(v4)
        f4 = F.interpolate(f4, size=(52, 52), mode='bilinear', align_corners=False)  # 上采樣到 52x52
        f4 = self.f2_cat(torch.cat([f4, f5], dim=1))
        
        # 融合 3：b, 256, 28, 28
        f3 = self.f3_v_proj(v3)
        f3 = F.interpolate(f3, size=(52, 52), mode='bilinear', align_corners=False)  # 上采樣到 52x52
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))
        
        # 聚合：將 f3, f4 和 f5 拼接在一起，並進行卷積
        fq = torch.cat([f3, f4, f5], dim=1)  # 確保拼接後的通道數為 1792
        fq = self.aggr(fq)
        fq = self.coordconv(fq)
        # b, 512, 52, 52
        return fq

# ---------- MBA Neck (Bi-direction Cross-Attention) ----------
class MultiModalBlock(nn.Module):
    """
    一層最小 MBA：Image->Text  &  Text->Image  交叉注意力
    """
    def __init__(self, v_dim, t_dim, n_heads=8):
        super().__init__()
        self.v2t_attn = nn.MultiheadAttention(embed_dim=t_dim, num_heads=n_heads)
        self.t2v_attn = nn.MultiheadAttention(embed_dim=v_dim, num_heads=n_heads)
        self.v_ln = nn.LayerNorm(v_dim)
        self.t_ln = nn.LayerNorm(t_dim)

    def forward(self, v_feat, t_tok):
        """
        v_feat : (B, C, H, W)
        t_tok  : (B, L, C_t)
        return : 更新後 v_feat , t_tok
        """
        B, C, H, W = v_feat.shape
        L = t_tok.size(1)
        # --- Image→Text ---
        t_in = self.t_ln(t_tok)                   # (B,L,C_t)
        v_flat = v_feat.flatten(2).permute(2,0,1) # (HW,B,C)
        t_q = t_in.permute(1,0,2)                 # (L,B,C_t)
        t_ctx, _ = self.v2t_attn(t_q, v_flat, v_flat)
        t_tok = t_tok + t_ctx.permute(1,0,2)

        # --- Text→Image ---
        v_in = self.v_ln(v_feat.flatten(2).permute(2,0,1))  # (HW,B,C)
        t_kv = t_tok.permute(1,0,2)                         # (L,B,C_t)
        v_ctx, _ = self.t2v_attn(v_in, t_kv, t_kv)
        v_feat = v_feat + v_ctx.permute(1,2,0).view(B, C, H, W)

        return v_feat, t_tok


class MBANeck(nn.Module):
    """
    替代原 FPN 的 Multi-scale Bi-direction Attention Neck
    in_channels     : backbone 3 個 stage 的通道
    embed_dim       : token / 最深視覺特徵通道 (預設 512)
    """
    def __init__(self,
                 in_channels=[128, 256, 512],
                 embed_dim=512,
                 n_heads=8):
        super().__init__()

        # 把 backbone 每層通道轉成同一維度 (embed_dim)
        self.v_proj = nn.ModuleList([
            nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_channels[::-1]
        ])  # 先處理 v5, v4, v3 的順序

        self.t_proj = nn.Linear(embed_dim, embed_dim)

        # 3 個 MultiModalBlock（對應 backbone 3 層）
        self.mba = nn.ModuleList([
            MultiModalBlock(embed_dim, embed_dim, n_heads) for _ in range(3)
        ])

        # 聚合成輸出 256 channels (與舊 FPN 對齊)
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.GELU()
        )

    def forward(self, imgs, txt_tok):
        """
        imgs    : list[ v3 , v4 , v5 ]，各 shape=(B,C_i,H_i,W_i)  (注意順序!)
        txt_tok : (B, L, embed_dim)
        return  : fq  (B,256,52,52)  與舊 FPN 相同
        """
        v_feats = imgs[::-1]          # 變成 [v5,v4,v3]
        B, _, _, _ = v_feats[0].shape
        t_feat = self.t_proj(txt_tok) # (B,L,embed_dim)

        agg = None
        for i, (v, proj, mba) in enumerate(zip(v_feats, self.v_proj, self.mba)):
            v = proj(v)                       # 換通道
            v, t_feat = mba(v, t_feat)        # 雙向 attention

            # 上採到 52×52 後累加
            v_up = F.interpolate(v, size=(52, 52), mode='bilinear', align_corners=False)
            agg = v_up if agg is None else agg + v_up

        fq = self.out_conv(agg)               # b,256,52,52
        return fq
