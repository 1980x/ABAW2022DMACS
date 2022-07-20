'''
Aum Sri Sai Ram

ABAW4 for MTL

Code to get predictions on test set by loading the checkpoints.

Authors: Darshan Gera, Badveeti Naveen Siva Kumar, Bobbili Veerendra Raj Kumar, Dr. S. Balasubramanian, SSSIHL

Date: 20-07-2022

Email: darshangera@sssihl.edu.in

'''

from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math
from datetime import datetime
import pytz

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
 
from models.backbone import ResNet_18
import dataset.abaw as dataset

from losses import classification_loss_func, regression_loss_func, cross_entropy_loss, CCCLoss, symmetric_kl_div


from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, EXPR_metric, VA_metric, AU_metric

parser = argparse.ArgumentParser(description='Test Code')
# Optimization options
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
                        
# Miscs
parser.add_argument('--manualSeed', type=int, default=5, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0,1,2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')



#Data
parser.add_argument('--test-root', type=str, default='../data/Affwild2/cropped_aligned_2022_MTL_test',
                        help="root path to train data directory")
                        
parser.add_argument('--label-test', default='../data/Affwild2/MTL_Challenge_test_set_release.txt', type=str, help='')

parser.add_argument('--chkpath', default='', type=str, help='path to checkpoint on which to run the test set')

parser.add_argument('--num_exp_classes', type=int, default=8, help='number of expression classes')
parser.add_argument('--num_aus', type=int, default=12, help='number of facial action units')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)



def main():
    
    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing DB')
    mean=[0.485, 0.456, 0.406]
    std =[0.229, 0.224, 0.225]
	
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    test_set = dataset.Dataset_ABAW_test(args.test_root, args.label_test, transform=transform_test)

    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.num_workers)
    print('Test dataset size: ', len(test_set))
    
    # Model
    print("==> creating ResNet-18")

    def create_model(ema=False):
        model = ResNet_18()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(chkpath)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    
    cudnn.benchmark = True
        
    #Validating the model
    test(test_loader, model, use_cuda, mode='Test Stats')#3 <- (4)

def test(dataloader, model, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    
    args.validate_iteration = int(len(dataloader.dataset)/args.batch_size)

    print('\n')
    outpath = (os.path.join(args.out, 'Preds_with_ssl_reweights_{}'.format(datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d-%m-%Y_%H-%M-%S'))))

    fout = open(outpath, 'w')
    print('image,valence,arousal,expression,aus', file=fout)
    
    bar = Bar('Test', max=args.validate_iteration)
    
    labeled_val_iter = iter(dataloader)    
    model.eval()
         
    with torch.no_grad():
        for batch_idx in range(args.validate_iteration):
            try:

               inputs, img_path = labeled_val_iter.next()
               
            except:
               labeled_val_iter = iter(dataloader)
               
               inputs, img_path = labeled_val_iter.next()

            
            # measure data loading time
            data_time.update(time.time() - end)       
            
            if mode == 'Train Stats':
                inputs = inputs[0]
                
            batch_size = inputs.size(0)

            if use_cuda:
               inputs = inputs.cuda()

            outputs = model(inputs)

            outputs_exp = F.softmax(outputs['EXPR'] , dim=-1).argmax(-1).float()
            outputs_au = (torch.sigmoid(outputs['AU']) > 0.5).float()
            outputs_va = torch.clamp(outputs['VA'], min=-1.0, max=1.0)
            
            for image in range(batch_size):
                out_v = '{0:.3f}'.format(outputs_va[image,0])
                out_a = '{0:.3f}'.format(outputs_va[image,1])
                out_exp = str(int(outputs_exp[image]))
                out_au = ''
                for i in range(11):
                    out_au += str(int(outputs_au[image][i]))
                    out_au += (',')
                out_au += str(int(outputs_au[image][11]))
                s = (img_path[image] + ',' + out_v + ',' + out_a + ','+ out_exp + ','+ out_au)	
                print(s, file=fout)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # plot progress
            bar.suffix  = '(({batch}/{size})Total:{total:}'.format(
                       batch=batch_idx + 1,
                       size=args.validate_iteration,
                       total=bar.elapsed_td
                       )
            bar.next()
    
    bar.finish()
    fout.close()

    
if __name__ == '__main__':
    main()  
