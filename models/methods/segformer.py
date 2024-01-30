import torch
from torch import nn
import torch.nn.functional as F

from ..ops import ConvBNReLU


class SegFormer(nn.Module):
    def __init__(self, backbone_channels, out_channels, align_corners):
        super().__init__()

        self.convs = nn.ModuleList()
        for in_channels in backbone_channels:
            self.convs.append(ConvBNReLU(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=1))

        self.fusion_conv = ConvBNReLU(in_channels=out_channels * len(backbone_channels),
                                      out_channels=out_channels,
                                      kernel_size=1)

        self.align_corners = align_corners


    def forward(self, *features):
        features = features[2:]

        outs = []
        for idx in range(len(features)):
            x = self.convs[idx](features[idx])
            x = F.interpolate(input=x,
                              size=features[0].shape[2:],
                              mode="bilinear",
                              align_corners=self.align_corners)
            outs.append(x)

        outs = self.fusion_conv(torch.cat(outs, dim=1))

        return outs
