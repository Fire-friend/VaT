import torchvision

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import numbers

from torch.nn import init


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################

class Model(nn.Module):
    def __init__(self, opt=None, input_channel=3):
        base_dim = 16
        super(Model, self).__init__()
        self.in_ch = input_channel
        self.mixInput = nn.Sequential(
            nn.Conv2d(input_channel, base_dim, kernel_size=3, padding=1, stride=1),
            TransformerBlock(base_dim, 2, 2.66, False, 'WithBias'),
            nn.Conv2d(base_dim, 3, kernel_size=1, padding=0, stride=1),
        )
        if self.in_ch == 6:
            pass
        self.proj = nn.Conv2d(3, base_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.down1 = Downsample(base_dim)
        self.trans_down1 = TransformerBlock(base_dim * 2, 2, 2.66, False, 'WithBias')
        self.down2 = Downsample(base_dim * 2)
        self.trans_down2 = TransformerBlock(base_dim * 2 ** 2, 2, 2.66, False, 'WithBias')
        self.up1 = Upsample(base_dim * 2 ** 2)
        self.reduce_chan1 = nn.Conv2d(base_dim * 2 ** 2, base_dim * 2, kernel_size=1)
        self.trans_up3 = TransformerBlock(base_dim * 2, 2, 2.66, False, 'WithBias')
        self.up2 = Upsample(base_dim * 2)
        self.out = nn.Conv2d(base_dim, 3, kernel_size=3, stride=1, padding=1, bias=True)
        # self._initialize_weights()

    def _initialize_weights(self):
        # Initialize convolution layers with zero weights but add small perturbation for gradients
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.constant_(m.weight, 1e-4)
                # if m.bias is not None:
                #     torch.nn.init.constant_(m.bias, 1e-5)
                # # Add small random noise (perturbation) to break symmetry and ensure gradients flow
                # with torch.no_grad():
                #     m.weight.add_(torch.randn_like(m.weight) * 1e-5)  # Small perturbation

    def forward(self, LLm_out, skip=False, only_merge=False):  # , LLm_input
        # LLm_res = LLm_out - LLm_input
        if self.in_ch == 6 and skip == False:
            weight = torch.sigmoid(self.mixInput(LLm_out))
            # weight = torch.zeros_like(self.mixInput(LLm_out))
            x = (1-weight) * LLm_out[:, :3] + weight * LLm_out[:, 3:]
            # x = weight
        else:
            x = LLm_out
        if only_merge:
            return x
        proj = self.proj(x)
        proj_downx2 = self.down1(proj)
        proj_downx2_tran = self.trans_down1(proj_downx2)
        proj_downx4 = self.down2(proj_downx2_tran)
        proj_downx4_tran = self.trans_down2(proj_downx4)
        proj_down_x2_up = self.up1(proj_downx4_tran)
        proj_cat_x2 = self.reduce_chan1(torch.cat([proj_down_x2_up, proj_downx2_tran], dim=1))
        proj_cat_x2_tran = self.trans_up3(proj_cat_x2)
        proj_cat_up = self.up2(proj_cat_x2_tran)
        out = self.out(proj_cat_up)# + x

        if self.in_ch == 6 and skip == False:
            return out, x, weight
        else:
            return out
