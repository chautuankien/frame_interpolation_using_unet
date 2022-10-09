import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import my_models.resnet as models
from my_models.Synthesis.synthesisModule import SynthesisModule
from my_models.MultiScaleModule import aspp, rfbNet

__all__ = ('basic_dvf')


def meshgrid(height, width):
    x_t = torch.matmul(
        torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width))
    y_t = torch.matmul(
        torch.linspace(-1.0, 1.0, height).view(height, 1), torch.ones(1, width))

    grid_x = x_t.view(1, height, width)
    grid_y = y_t.view(1, height, width)
    return grid_x, grid_y



class DVF_Net(nn.Module):
    """
    Basic Deep Voxel Flow Network
    """
    def __init__(self):
        super(DVF_Net, self).__init__()

        nb_filter = [64, 128, 256, 512]

        self.conv1 = ConvBlock(6, nb_filter[0])
        self.conv2 = Down(nb_filter[0], nb_filter[1])
        self.conv3 = Down(nb_filter[1], nb_filter[2])
        self.conv4 = Down(nb_filter[2], nb_filter[3])

        rates = (1, 6, 12, 18)
        self.aspp = aspp.ASPP(nb_filter[3], int(nb_filter[3] / len(rates)), rates)
        # self.rfb = rfbNet.BasicRFB(nb_filter[3], nb_filter[3], stride=1, scale=1.0, visual=2)

        self.deconv1 = Up(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.deconv2 = Up(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.deconv3 = Up(nb_filter[0] + nb_filter[1], nb_filter[0])

        self.output = OutConv(nb_filter[0], 3)

        self.synthesis = SynthesisModule()


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x_multiscale = self.aspp(x4)

        de_x1 = self.deconv1(x_multiscale, x3)
        de_x2 = self.deconv2(de_x1, x2)
        de_x3 = self.deconv3(de_x2, x1)

        output = self.output(de_x3)

        final = self.synthesis(output, x)

        return final

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.conv_block(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return  self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)

        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.output = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Tanh()
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.output(x)

def basic_dvf():
    model = DVF_Net()
    return model




