import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import ConvBNReLU


class UNet(nn.Module):
    def __init__(self, backbone_channels, out_channels, align_corners):
        super().__init__()

        backbone_channels = backbone_channels[::-1]
        out_channels = [out_channels * i for i in [16, 8, 4, 2, 1]]
        in_channels = [backbone_channels[0]] + list(out_channels[:-1])
        skip_channels = list(backbone_channels[1:]) + [0]

        blocks = [Block(in_channels=in_ch,
                        skip_channels=skip_ch,
                        out_channels=out_ch,
                        align_corners=align_corners)
                  for in_ch, skip_ch, out_ch
                  in zip(in_channels, skip_channels, out_channels)]
        self.blocks = nn.ModuleList(blocks)


    def forward(self, *features):
        features = features[1:][::-1]

        outs = features[0]
        skips = features[1:]

        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            outs = block(outs, skip)

        return outs


class Block(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, align_corners):
        super().__init__()

        self.conv = nn.Sequential(ConvBNReLU(in_channels=in_channels + skip_channels,
                                             out_channels=out_channels,
                                             padding=1),
                                  ConvBNReLU(in_channels=out_channels,
                                             out_channels=out_channels,
                                             padding=1))

        self.align_corners = align_corners


    def forward(self, x, skip):
        size = [s * 2 - int(self.align_corners) for s in x.shape[2:]]
        outs = F.interpolate(input=x,
                             size=size,
                             mode="bilinear",
                             align_corners=self.align_corners)

        if skip is not None:
            outs = torch.cat([outs, skip], dim=1)

        outs = self.conv(outs)

        return outs
