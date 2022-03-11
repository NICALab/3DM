""" Full assembly of the parts to form the complete network """
import torch.nn as nn

import sys
sys.path.append(".")
from codes.unet3d.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, 
                 depth, channel,
                 normalization=None,
                 activation='lrelu',
                 bias=True):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth
        self.channel = channel

        if type(channel) is int:
            channels = []
            for i in range(depth + 1):
                channels.append(channel * pow(2, i))
        
        self.incblock = []
        self.incblock.append(Conv(n_channels, channels[0], kernel_size=3, bias=bias,
                                        normalization=None,
                                        activation=activation))

        z_true_idx = 0
        if depth > 1:
            for i in range(depth - 1):
                self.incblock.append(Down(channels[i], channels[i+1],
                                          kernel_size=3, bias=True,
                                          normalization=normalization,
                                          activation=activation,
                                          z_true=bool(z_true_idx)))
                z_true_idx = 1 - z_true_idx
        self.incblock.append(Down(channels[-2], channels[-1],
                                  kernel_size=3, bias=bias,
                                  normalization=normalization,
                                  activation=activation,
                                  z_true=bool(z_true_idx)))
        self.incblockMod = nn.ModuleList(self.incblock)

        self.decblock = []
        if depth > 1:
            for i in range(depth - 1):
                self.decblock.append(Up(channels[-i-1], channels[-i-2],
                                        kernel_size=3, bias=bias,
                                        normalization=normalization,
                                        activation=activation,
                                        z_true=bool(z_true_idx)))
                z_true_idx = 1 - z_true_idx
        self.decblock.append(Up(channels[1], channels[0],
                                kernel_size=3, bias=bias,
                                normalization=normalization,
                                activation=activation,
                                z_true=bool(z_true_idx)))
        self.decblock.append(OutConv(channels[0], n_classes,
                                     kernel_size=3, bias=None,
                                     normalization=normalization,
                                     activation=activation))

        self.decblockMod = nn.ModuleList(self.decblock)

    def forward(self, x):
        # x, pads = self.pad_to(x, int(math.pow(2, self.depth)))

        xs = []
        xs.append(x)
        for mod in self.incblock:
            x = mod(x)
            xs.append(x)

        x = self.decblock[0](xs[-1], xs[-2])

        for i in range(self.depth - 1):
            x = self.decblock[i + 1](x, xs[-3-i])
        output = self.decblock[-1](x)

        return output



if __name__ == "__main__":
    net = UNet(n_channels=1, n_classes=1, depth=4, channel=1,
               normalization=None, activation='lrelu').cuda()
    import torchsummary

    torchsummary.summary(net, (1, 128, 128, 64))
    print(net)