import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class VGG_Loss(nn.Module):
    def __init__(self):
        super(VGG_Loss, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
        for param in self.vgg16_conv_4_3.parameters():
            param.requires_grad = False

    def forward(self, output, gt):
        vgg_output = self.vgg16_conv_4_3(output)
        with torch.no_grad():
            vgg_gt = self.vgg16_conv_4_3(gt.detach())

        loss = F.mse_loss(vgg_output, vgg_gt)

        return loss



class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_name = loss.split('*')
            if loss_name == 'L1':
                loss_function = nn.L1Loss()
            elif loss_name == 'VGG':
                loss_function = VGG_Loss()
            elif loss_name == 'MSE':
                loss_function = nn.MSELoss()
            self.loss.append({
                'name': loss_name,
                'weight': float(weight),
                'function': loss_function
            })

        if len(self.loss) > 1:
            self.loss.append({'name': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                self.loss_module.append(l['function'])

        device = torch.device('cuda' if args.cuda else 'cpu')
        self.loss_module.to(device)

    def forward(self, output, gt):
        loss = 0
        losses = {}
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                _loss = l['function'](output, gt)
                effective_loss = l['weight'] * _loss
                losses[l['name']] = effective_loss
                loss += effective_loss

        return loss, losses