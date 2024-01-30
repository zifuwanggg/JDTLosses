import torch
from torch import nn
import torch.nn.functional as F

from ..ops import ConvBNReLU, DWConvBNReLU


class DeepLabV3(nn.Module):
    def __init__(self,
                 backbone_channels,
                 out_channels,
                 atrous_rates,
                 align_corners):
        super().__init__()

        self.aspp = ASPP(in_channels=backbone_channels[-1],
                         out_channels=out_channels,
                         atrous_rates=atrous_rates,
                         align_corners=align_corners,
                         DW=False)

        self.bottleneck = ConvBNReLU(in_channels=(len(atrous_rates) + 2) * out_channels,
                                     out_channels=out_channels,
                                     padding=1)


    def forward(self, *features):
        outs = self.aspp(features[-1])
        outs = self.bottleneck(outs)

        return outs


class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 atrous_rates,
                 align_corners,
                 DW=False):
        super(ASPP, self).__init__()

        if DW:
            ASPPConv = DWConvBNReLU
        else:
            ASPPConv = ConvBNReLU

        modules = []
        modules.append(ASPPPooling(in_channels=in_channels,
                                   out_channels=out_channels,
                                   align_corners=align_corners))
        modules.append(ASPPConv(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                padding=0,
                                dilation=1))
        for dilation in atrous_rates:
            modules.append(ASPPConv(in_channels=in_channels,
                                    out_channels=out_channels,
                                    padding=dilation,
                                    dilation=dilation))

        self.convs = nn.ModuleList(modules)


    def forward(self, x):
        outs = []
        for conv in self.convs:
            outs.append(conv(x))

        outs = torch.cat(outs, dim=1)

        return outs


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, align_corners):
        super().__init__(nn.AdaptiveAvgPool2d(1),
                         ConvBNReLU(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1))

        self.align_corners = align_corners


    def forward(self, x):
        size = x.shape[2:]

        for mod in self:
            x = mod(x)

        x = F.interpolate(input=x,
                          size=size,
                          mode="bilinear",
                          align_corners=self.align_corners)

        return x
