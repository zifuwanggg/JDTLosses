import torch
from torch import nn
import torch.nn.functional as F

from .deeplabv3 import ASPP
from ..ops import ConvBNReLU, DWConvBNReLU


class DeepLabV3Plus(nn.Module):
    def __init__(self,
                 backbone_channels,
                 out_channels,
                 shortcut_channels,
                 atrous_rates,
                 align_corners):
        super().__init__()

        self.aspp = ASPP(in_channels=backbone_channels[-1],
                         out_channels=out_channels,
                         atrous_rates=atrous_rates,
                         align_corners=align_corners,
                         DW=True)

        self.bottleneck = ConvBNReLU(in_channels=(len(atrous_rates) + 2) * out_channels,
                                     out_channels=out_channels,
                                     padding=1)

        self.shortcut = ConvBNReLU(in_channels=backbone_channels[-4],
                                   out_channels=shortcut_channels,
                                   kernel_size=1)

        self.cat_conv = nn.Sequential(DWConvBNReLU(in_channels=shortcut_channels + out_channels,
                                                   out_channels=out_channels,
                                                   padding=1),
                                      DWConvBNReLU(in_channels=out_channels,
                                                   out_channels=out_channels,
                                                   padding=1))

        self.align_corners = align_corners


    def forward(self, *features):
        aspp_features = self.aspp(features[-1])
        aspp_features = self.bottleneck(aspp_features)
        aspp_features = F.interpolate(input=aspp_features,
                                      size=features[-4].shape[2:],
                                      mode="bilinear",
                                      align_corners=self.align_corners)

        shortcut_features = self.shortcut(features[-4])

        cat_features = torch.cat([aspp_features, shortcut_features], dim=1)
        cat_features = self.cat_conv(cat_features)

        return cat_features
