import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Atrous_module(nn.Module):
    def __init__(self, inplanes, planes, rate, relu=True, bn=True):
        super(Atrous_module, self).__init__()
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=(3, 3),
                                            stride=(1, 1), padding=rate, dilation=rate)
        self.batch_norm = nn.BatchNorm2d(planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.atrous_convolution(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.relu is not None:
            x = self.relu(x)

        return x

class ASPP(nn.Module):
    def __init__(self, in_planes, out_planes, rates):
        super(ASPP, self).__init__()

        self.aspp1 = Atrous_module(in_planes, out_planes, rate=rates[0])
        self.aspp2 = Atrous_module(in_planes, out_planes, rate=rates[1])
        self.aspp3 = Atrous_module(in_planes, out_planes, rate=rates[2])
        self.aspp4 = Atrous_module(in_planes, out_planes, rate=rates[3])

        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                        nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1)),
                                        nn.BatchNorm2d(out_planes),
                                        nn.ReLU(inplace=True))

        self.bottleneck = nn.Sequential(nn.Conv2d(out_planes * 5, out_planes * 4, kernel_size=(1, 1), bias=False),
                                        nn.BatchNorm2d(out_planes * 4),
                                        nn.Dropout(0.3),
                                        nn.ReLU(inplace=True))
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
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        x5 = self.image_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.bottleneck(x)

        return x

if __name__ == '__main__':
    model = ASPP(1024, 256, [1, 6, 12, 18])
    print(model)
    summary(model.cuda(), (1024, 256, 256))


