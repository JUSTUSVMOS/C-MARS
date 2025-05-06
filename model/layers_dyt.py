import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 這裡先定義幾個 DyT 替代模塊，用於取代原本的 BatchNorm / LayerNorm

class DyT2d(nn.Module):
    """
    取代 nn.BatchNorm2d 的 DyT 實現 (2D)
    """
    def __init__(self, num_features, alpha_init=0.5):
        super().__init__()
        # learnable scalar
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        # learnable affine
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # x: [N, C, H, W]
        # tanh(alpha * x)
        out = torch.tanh(self.alpha * x)
        # channel-wise affine
        out = out * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return out


class DyT1d(nn.Module):
    """
    取代 nn.BatchNorm1d 的 DyT 實現 (1D)
    適用於線性層的輸出 (N, C) 或 (N, C, L) 等等
    """
    def __init__(self, num_features, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # x: [N, C] 或 [N, L, C], 這裡以 [N, C] 為主
        out = torch.tanh(self.alpha * x)
        # 廣播到最後一維做通道仿射
        if out.dim() == 2:
            # [N, C]
            out = out * self.gamma + self.beta
        else:
            # 若是 [N, L, C]，則將 gamma/beta broadcast 到最後一維
            out = out * self.gamma + self.beta
        return out


class DyTLayerNorm(nn.Module):
    """
    取代 nn.LayerNorm 的 DyT 實現
    用於 Transformer Decoder Layer 等需 LayerNorm 的地方
    """
    def __init__(self, normalized_shape, alpha_init=0.5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init)
        self.gamma = nn.Parameter(torch.ones(*normalized_shape))
        self.beta = nn.Parameter(torch.zeros(*normalized_shape))

    def forward(self, x):
        # x 的形狀一般是 [N, ..., normalized_shape]，這裡直接做 element-wise
        out = torch.tanh(self.alpha * x)
        # 底下做廣播
        return out * self.gamma + self.beta


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        # 將原本的 nn.BatchNorm2d 改為 DyT2d
        DyT2d(out_dim),
        nn.GELU())


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias),
        # 將原本的 nn.BatchNorm1d 改為 DyT1d
        DyT1d(out_dim),
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
        # 將原本的 nn.LayerNorm(d_model) 替換為 DyTLayerNorm(d_model)
        self.norm = DyTLayerNorm(d_model)
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
        # 將原本的 nn.LayerNorm(d_model) 都改成 DyTLayerNorm(d_model)
        self.self_attn_norm = DyTLayerNorm(d_model)
        self.cross_attn_norm = DyTLayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            # 這裡原本是 nn.LayerNorm(dim_feedforward)，也改為 DyTLayerNorm
            DyTLayerNorm(dim_feedforward),
            nn.Linear(dim_feedforward, d_model)
        )
        # LayerNorm & Dropout (全部改成 DyTLayerNorm)
        self.norm1 = DyTLayerNorm(d_model)
        self.norm2 = DyTLayerNorm(d_model)
        self.norm3 = DyTLayerNorm(d_model)
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
        # 將原本的 nn.BatchNorm2d 改成 DyT2d
        self.norm_layer = nn.Sequential(
            DyT2d(out_channels[2]),
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
        # 將原本的 nn.BatchNorm2d 改成 DyT2d
        self.coordconv = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[1], kernel_size=3, padding=1),
            DyT2d(out_channels[1]),
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
