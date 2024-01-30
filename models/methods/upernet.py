import torch
from torch import nn
import torch.nn.functional as F

from .pspnet import PPM
from ..ops import ConvBNReLU


class UPerNet(nn.Module):
    def __init__(self, backbone_channels, out_channels, bins, align_corners):
        super().__init__()

        self.ppm = PPM(in_channels=backbone_channels[-1],
                       out_channels=out_channels,
                       bins=bins,
                       align_corners=align_corners)

        self.bottleneck = ConvBNReLU(in_channels=backbone_channels[-1] + len(bins) * out_channels,
                                     out_channels=out_channels,
                                     padding=1)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in backbone_channels[:-1]:
            lateral_conv = ConvBNReLU(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=1)
            fpn_conv = ConvBNReLU(in_channels=out_channels,
                                  out_channels=out_channels,
                                  padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvBNReLU(in_channels=len(backbone_channels) * out_channels,
                                         out_channels=out_channels,
                                         padding=1)

        self.align_corners = align_corners


    def forward(self, *features):
        features = features[2:]
        laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        psp_outs = self.ppm(features[-1])
        psp_outs = self.bottleneck(psp_outs)
        laterals.append(psp_outs)

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(input=laterals[i],
                                                              size=laterals[i - 1].shape[2:],
                                                              mode="bilinear",
                                                              align_corners=self.align_corners)

        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(input=fpn_outs[i],
                                        size=fpn_outs[0].shape[2:],
                                        mode='bilinear',
                                        align_corners=self.align_corners)

        fpn_outs = torch.cat(fpn_outs, dim=1)
        fpn_outs = self.fpn_bottleneck(fpn_outs)

        return fpn_outs
