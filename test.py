import os
import sys
import time
import copy
import shutil
import random
import pdb

import torch
import numpy as np
from tqdm import tqdm

from torchvision.utils import save_image
import config
import utils
import math
import torch.nn.functional as F

from torch.utils.data import DataLoader

##### Parse CmdLine Arguments #####
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
args, unparsed = config.get_args()
cwd = os.getcwd()

device = torch.device('cuda' if args.cuda else 'cpu')

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

#### Dataset loading ####
if args.datasetName == 'UCF101':
    from datasets.ucf101 import ucf101
    test_set = ucf101.UCF101(root=args.datasetPath + 'test4', is_training=False)
elif args.datasetName == 'Vimeo_90K':
    from datasets.vimeo_90K.vimeo_90K import Vimeo_90K
    test_set = Vimeo_90K(root=args.datasetPath, is_training=False)
elif args.datasetName == 'VimeoSepTuplet':
    from datasets.vimeo_90K.vimeo_90K import VimeoSepTuplet
    test_set = VimeoSepTuplet(root=args.datasetPath, is_training=False, mode='mini')

test_loader = torch.utils.data.DataLoader(test_set, pin_memory=True if args.cuda else False,
                                             batch_size=args.test_batch_size,
                                             num_workers=args.num_workers,
                                             shuffle=False, drop_last=True)

print("Building model: %s"%args.modelName)
if args.modelName == 'basic_dvf':
    from my_models.DVF.dvf_models import DVF_Net
    model = DVF_Net()
else:
    raise NotImplementedError('Model is not implemented')

#model = torch.nn.DataParallel(model).to(device)
model = model.to(device)
print("#params", sum([p.numel() for p in model.parameters()]))

def save_batch_images(ims_pred, ims_gt):
    if not os.path.exists('./test_results'):
        os.makedirs('./test_results')
    # Save every image in batch to indicated location
    for j in range(ims_pred.size(0)):
        pred_name = str(args.out_counter) + '_out.png'
        gt_name = str(args.out_counter) + '_gt.png'

        save_image(ims_pred[j, :, :, :], os.path.join('./test_results', pred_name))
        save_image(ims_gt[j, :, :, :], os.path.join('./test_results', gt_name))

        args.out_counter += 1

def to_psnr(rect, gt):
    mse = F.mse_loss(rect, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    psnr_list = [-10.0 * math.log10(mse) for mse in mse_list]
    return psnr_list

def test(args):
    time_taken = []
    losses, psnrs, ssims = utils.init_meters(args.loss,reset_loss=True)
    model.eval()
    args.out_counter = 0

    start = time.time()
    with torch.no_grad():
        for i, (images, gt_image) in enumerate(tqdm(test_loader)):

            images = [img_.cuda() for img_ in images]
            gt = gt_image.cuda()

            torch.cuda.synchronize()

            out = model(images)

            torch.cuda.synchronize()
            time_taken.append(time.time() - start)

            utils.eval_metrics(out, gt, psnrs, ssims)

            save_batch_images(out, gt)

    print("PSNR: %f, SSIM: %fn" %(psnrs.avg, ssims.avg.item()))
    print("Time , ", sum(time_taken)/len(time_taken))

    return psnrs.avg


""" Entry Point """
def main(args):
    
    assert args.checkpoint_dir is not None
    checkpoint = torch.load(args.checkpoint_dir)

    model_dict = model.state_dict()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    test(args)


if __name__ == "__main__":
    main(args)
