from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        super().__init__(nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   bias=bias),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


class DWConvBNReLU(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        super().__init__(nn.Conv2d(in_channels=in_channels,
                                   out_channels=in_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=in_channels,
                                   bias=bias),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))


class SegHead(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout,
                 crop_size,
                 align_corners):
        super().__init__(nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
                         nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1),
                         nn.Upsample(size=crop_size,
                                     mode="bilinear",
                                     align_corners=align_corners))
