import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

        # self.bottleneck = nn.Conv2d(in_dim + reduction_dim*4, in_dim, kernel_size=1, bias=False)

    def forward(self, x):
        x_size = x.size()
        out = [x]

        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))

        # bottleneck = self.bottleneck(torch.cat(out, 1))
        return torch.cat(out, 1)

if __name__ == '__main__':
    model = PPM(1024, 256, (1, 2, 3, 6))
    print(model)
    # summary(model.cuda(), ())

