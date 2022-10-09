import torch
import torch.nn as nn

#from config import args

__all__ = ['SynthesisModule']

class SynthesisModule(nn.Module):
    def __init__(self):
        super(SynthesisModule, self).__init__()

    def forward(self, voxelflow, input):
        flow = voxelflow[:, 0:2, :, :]
        mask = voxelflow[:, 2:3, :, :]

        grid_x, grid_y = meshgrid(224, 224)
        with torch.cuda.device(input.get_device()):
            grid_x = torch.autograd.Variable(
                grid_x.repeat([input.size()[0], 1, 1])).cuda()
            grid_y = torch.autograd.Variable(
                grid_y.repeat([input.size()[0], 1, 1])).cuda()

        flow = 0.5 * flow

        # Interpolation method
        coor_x_1 = grid_x - flow[:, 0, :, :]
        coor_y_1 = grid_y - flow[:, 1, :, :]
        coor_x_2 = grid_x + flow[:, 0, :, :]
        coor_y_2 = grid_y + flow[:, 1, :, :]

        output_1 = torch.nn.functional.grid_sample(
            input[:, 0:3, :, :],
            torch.stack([coor_x_1, coor_y_1], dim=3),
            padding_mode='border', align_corners=True)
        output_2 = torch.nn.functional.grid_sample(
            input[:, 3:6, :, :],
            torch.stack([coor_x_2, coor_y_2], dim=3),
            padding_mode='border', align_corners=True)
        mask = 0.5 * (1.0 + mask)
        mask = mask.repeat([1, 3, 1, 1])
        x = mask * output_1 + (1.0 - mask) * output_2

        return x

def meshgrid(height, width):
    x_t = torch.matmul(
        torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width))
    y_t = torch.matmul(
        torch.linspace(-1.0, 1.0, height).view(height, 1), torch.ones(1, width))

    grid_x = x_t.view(1, height, width)
    grid_y = y_t.view(1, height, width)
    return grid_x, grid_y