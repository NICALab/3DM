""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size, bias,
                 normalization, activation, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = (in_channels + 1) // 2
        else:
            mid_channels = mid_channels
        # print\('bias : ', bias)
        padding = kernel_size // 2
        conv_list = []
        conv_list.append(nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=bias))
        if normalization == 'BN':
            conv_list.append(nn.BatchNorm3d(mid_channels))
        elif normalization == 'IN':
            conv_list.append(nn.InstanceNorm3d(mid_channels))
        else:
            pass
        if activation == 'relu':
            conv_list.append(nn.ReLU())
        elif activation == 'lrelu':
            conv_list.append(nn.LeakyReLU(0.2))
        else:
            pass
        conv_list.append(nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias))
        if normalization == 'BN':
            conv_list.append(nn.BatchNorm3d(out_channels))
        elif normalization == 'IN':
            conv_list.append(nn.InstanceNorm3d(out_channels))
        else:
            pass
        if activation == 'relu':
            conv_list.append(nn.ReLU())
        elif activation == 'lrelu':
            conv_list.append(nn.LeakyReLU(0.2))
        else:
            pass

        self.conv = nn.Sequential(*conv_list)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, bias,
                 normalization, activation, z_true=True):
        super().__init__()
        module_list = []
        if z_true:
            module_list.append(nn.MaxPool3d((2, 2, 2)))
        else:
            module_list.append(nn.MaxPool3d((1, 2, 2)))
        module_list.append(Conv(in_channels, out_channels, bias=bias, kernel_size=kernel_size,
                 normalization=normalization, activation=activation, mid_channels=in_channels))
        self.maxpool_conv = nn.Sequential(*module_list)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, bias,
                 normalization, activation, z_true=True):
        super().__init__()
        if z_true:
            self.up = nn.ConvTranspose3d(in_channels , in_channels // 2, bias=bias, kernel_size=2, stride=2)
        else:
            self.up = nn.ConvTranspose3d(in_channels , in_channels // 2, bias=bias, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = Conv(in_channels, out_channels, bias=bias, kernel_size=kernel_size,
                         normalization=normalization, activation=activation)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = F.upsample(x1, x2.size()[2:], mode='trilinear')
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias,
                 normalization, activation):
        super(OutConv, self).__init__()
        self.conv = Conv(in_channels, in_channels // 2, bias=bias, kernel_size=kernel_size,
                         normalization=normalization, activation=activation)
        self.outconv = nn.Conv3d(in_channels // 2, out_channels, bias=bias, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return self.outconv(x)