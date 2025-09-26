import torch.nn as nn
from mmcv.cnn import ConvModule
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from ..builder import NECKS

class EnhancedSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        return self.sigmoid(x)

class MultiScaleFeatureEnhancement(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)
        self.norm3 = nn.GroupNorm(32, channels)
        self.act = nn.ReLU(inplace=True)
        self.fusion = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        x1 = self.act(x1)
        
        x2 = self.conv2(x)
        x2 = self.norm2(x2)
        x2 = self.act(x2)
        
        x3 = self.conv3(x)
        x3 = self.norm3(x3)
        x3 = self.act(x3)
        
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.fusion(x)
        return x + identity

class CrossScaleFeatureFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(32, channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        # 确保x2的尺寸与x1匹配
        if x2.shape[2:] != x1.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = x1 + x2
        x = self.norm(x)
        x = self.act(x)
        return x

@NECKS.register_module()
class ImprovedChannelMapper(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 groups=1,
                 act_cfg=None,
                 norm_cfg=None,
                 num_outs=None):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        
        # 为每个输入通道创建对应的注意力模块
        self.channel_attentions = nn.ModuleList()
        self.spatial_attentions = nn.ModuleList()
        self.feature_enhancements = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.cross_fusions = nn.ModuleList()
        
        # 处理输入通道
        for in_channel in in_channels:
            self.channel_attentions.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channel, in_channel // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel // 16, in_channel, 1),
                nn.Sigmoid()
            ))
            self.spatial_attentions.append(EnhancedSpatialAttention())
            self.feature_enhancements.append(MultiScaleFeatureEnhancement(out_channels))
            self.convs.append(ConvModule(
                in_channel,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg))
            
        # 添加跨尺度特征融合
        for i in range(len(in_channels) - 1):
            self.cross_fusions.append(CrossScaleFeatureFusion(out_channels))
        
        # 添加额外的输出层
        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels), num_outs):
                self.extra_convs.append(ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg))

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert len(inputs) == len(self.in_channels)
        
        outs = []
        for i, x in enumerate(inputs):
            # 通道注意力
            ca = self.channel_attentions[i](x)
            x = x * ca
            # 空间注意力
            sa = self.spatial_attentions[i](x)
            x = x * sa
            # 特征转换
            x = self.convs[i](x)
            # 特征增强
            x = self.feature_enhancements[i](x)
            outs.append(x)
            
        # 跨尺度特征融合
        for i in range(len(outs) - 1):
            outs[i] = self.cross_fusions[i](outs[i], outs[i + 1])
            
        # 生成额外的输出层
        if self.num_outs > len(outs):
            for i in range(len(outs), self.num_outs):
                if i == len(outs):
                    x = outs[-1]
                else:
                    x = outs[-1]
                x = self.extra_convs[i - len(self.in_channels)](x)
                outs.append(x)
                
        return tuple(outs)