import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS


def _make_mlp(channels, reduction=16):
    # Shared MLP for channel attention
    return nn.Sequential(
        nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
    )


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = _make_mlp(channels, reduction)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg = F.adaptive_avg_pool2d(x, 1)
        max_ = F.adaptive_max_pool2d(x, 1)
        att = self.mlp(avg) + self.mlp(max_)
        return self.sigmoid(att)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, max_], dim=1)
        att = self.conv(cat)
        return self.sigmoid(att)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x) * x
        x = self.spatial_att(x) * x
        return x


@NECKS.register_module()
class CFPAN(BaseModule):
    """Feature Pyramid Network with Cross-Level Fusion Pyramid Attention."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(CFPAN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            self.add_extra_convs = 'on_input'

        # 1x1 lateral convolutions
        self.lateral_convs = nn.ModuleList()
        # 3x3 fpn convolutions
        self.fpn_convs = nn.ModuleList()
        for i in range(start_level, self.backbone_end_level):
            self.lateral_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False))
            self.fpn_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False))

        # CBAM attention at top level
        self.cbam = CBAM(out_channels)

        # Cross-level conv for raw backbone features
        self.cross_conv = nn.ModuleList()
        for i in range(start_level, self.backbone_end_level):
            self.cross_conv.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False))

        # Spatial weight generators for fusion (3 weights)
        used_levels = self.backbone_end_level - self.start_level
        self.weight_conv = nn.ModuleList()
        for _ in range(used_levels - 1):
            self.weight_conv.append(nn.Conv2d(out_channels * 3, 3, kernel_size=1))

        # extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - used_levels
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_c = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_c = out_channels
                self.fpn_convs.append(
                    ConvModule(
                        in_c,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False))

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        # bottom-up lateral features
        laterals = [l(conv_input) for l, conv_input in zip(self.lateral_convs, inputs[self.start_level:self.backbone_end_level])]

        # apply CBAM on highest-level
        laterals[-1] = self.cbam(laterals[-1])

        # initialize pyramid
        used = len(laterals)
        ps = [None] * used
        ps[-1] = laterals[-1]

        # cross-level fusion
        for idx, lvl in enumerate(range(used - 2, -1, -1)):
            # P_{n+1}
            p_up = F.interpolate(
                ps[lvl + 1], size=laterals[lvl].shape[2:], **self.upsample_cfg)
            # Conv(C_{n+1})
            cnp1 = inputs[lvl + 1 + self.start_level]
            cnp1_conv = self.cross_conv[lvl + 1](cnp1)
            cnp1_up = F.interpolate(cnp1_conv, size=laterals[lvl].shape[2:], **self.upsample_cfg)
            # Conv(C_n)
            cn = inputs[lvl + self.start_level]
            cn_conv = self.cross_conv[lvl](cn)
            # generate spatial weights
            w_feats = torch.cat([p_up, cnp1_up, cn_conv], dim=1)
            w = self.weight_conv[idx](w_feats)
            w = F.softmax(w, dim=1)
            a, b, g = w[:, :1], w[:, 1:2], w[:, 2:3]
            # fuse features
            ps[lvl] = a * p_up + b * cnp1_up + g * cn_conv

        # build outputs with 3x3 conv
        outs = [self.fpn_convs[i](ps[i]) for i in range(used)]

        # extra levels
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for _ in range(self.num_outs - len(outs)):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used](extra_source))
                for i in range(used + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
