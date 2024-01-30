import torch
from torch import nn
import torch.nn.functional as F

from ..ops import ConvBNReLU


class PSPNet(nn.Module):
    def __init__(self, backbone_channels, out_channels, bins, align_corners):
        super().__init__()

        self.ppm = PPM(in_channels=backbone_channels[-1],
                       out_channels=out_channels,
                       bins=bins,
                       align_corners=align_corners)

        self.bottleneck = ConvBNReLU(in_channels=backbone_channels[-1] + len(bins) * out_channels,
                                     out_channels=out_channels,
                                     padding=1)


    def forward(self, *features):
        outs = self.ppm(features[-1])
        outs = self.bottleneck(outs)

        return outs


class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, bins, align_corners):
        super(PPM, self).__init__()

        self.pools = nn.ModuleList()
        for bin in bins:
            self.pools.append(nn.Sequential(nn.AdaptiveAvgPool2d(bin),
                                            ConvBNReLU(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=1)))

        self.align_corners = align_corners


    def forward(self, x):
        outs = [x]
        for pool in self.pools:
            outs.append(F.interpolate(input=pool(x),
                                      size=x.shape[2:],
                                      mode="bilinear",
                                      align_corners=self.align_corners))

        outs = torch.cat(outs, 1)

        return outs
