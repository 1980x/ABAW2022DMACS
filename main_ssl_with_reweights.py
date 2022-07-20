'''
Aum Sri Sai Ram

ABAW4 for MTL

Implementation of SS-MFAR

Authors: Darshan Gera, Badveeti Naveen Siva Kumar, Bobbili Veerendra Raj Kumar, Dr. S. Balasubramanian, SSSIHL

Date: 20-07-2022

Resnet18 with separate hidden layer for feature learning along with task specific classifers, 
using weighted loss function to take care of imbalance in dataset, pretraining on ms_celeb,
Using Semi Supervised Learning for improved performance.

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

parser = argparse.ArgumentParser(description='Training with Reweights using Semi Supervised Learning')
# Optimization options
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
                        
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=5, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0,1,2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
                    
parser.add_argument('--train-iteration', type=int, default=800,help='Number of iteration per epoch')
                    
parser.add_argument('--out', default='result',  help='Directory to output the result')

parser.add_argument('--model-dir','-m', default='', type=str,help='Path where checkpoints are loaded')

#Data
parser.add_argument('--train-root', type=str, default='../data/Affwild2/cropped_aligned',
                        help="root path to train data directory")
parser.add_argument('--val_root', type=str, default='../data/Affwild2/cropped_aligned',
                        help="root path to test data directory")
                        
parser.add_argument('--label-train', default='../data/Affwild2/training_set_annotations_22.txt', type=str, help='')
parser.add_argument('--label-val', default='../data/Affwild2/validation_set_annotations_22.txt', type=str, help='')

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

best_perf = 0  # best performance

def main():
    global best_perf

    if not os.path.isdir(args.out):
        mkdir_p(args.out)
    if not os.path.isdir(args.model_dir):
        mkdir_p(args.model_dir)

    # Data
    print(f'==> Preparing DB')
    mean=[0.485, 0.456, 0.406]
    std =[0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomApply([
            transforms.RandomCrop(224, padding=8)
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_set = dataset.Dataset_ABAW_Affwild2(args.train_root, args.label_train, transform=dataset.TransformTwice(transform_train))
    
    val_set = dataset.Dataset_ABAW_Affwild2(args.val_root, args.label_val, transform=transform_val)
    
    trainloader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    
    val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, num_workers=args.num_workers)
    
    print('Train dataset size: ', len(train_set))
    print('Val dataset size: ', len(val_set))
    
    # Model
    print("==> creating ResNet-18")

    def create_model(ema=False):
        model = ResNet_18()
        model = torch.nn.DataParallel(model).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))  
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logger = Logger(os.path.join(args.out, 'ssmfar_logs_{}'.format(datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d-%m-%Y_%H-%M-%S'))), title='ABAWAffwild2')
    logger.set_names(['Train Loss', 'Train acc', 'Val Loss', 'Val Acc.', 'Val F1-Exp', 'Val CCC-VA', 'Val F1-AU', 'Overall Perf.'])

    #Iniitalizing
    val_perf = []
    threshold = [0.8 for _ in range(args.num_exp_classes)]
    start_epoch = 1
    
    au_criterion, va_criterion , exp_criterion = classification_loss_func, CCCLoss(digitize_num=1), cross_entropy_loss
    
    # Train and val
    for epoch in range(start_epoch, args.epochs + 1):
        
        print('\nEpoch: [%d | %d] LR: %f Threshold=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]' % (epoch, args.epochs, state['lr'], threshold[0], threshold[1], threshold[2], threshold[3], threshold[4], threshold[5], threshold[6], threshold[7]))
        
        #Training the model  
        train_loss, train_acc, exp_f1, va_avg, au_f1  = train(trainloader, model, optimizer,  au_criterion, va_criterion , exp_criterion, threshold, epoch, use_cuda)
        
        #Validate on the train set and get the outputs, targets(ground truth) to set the class specific thresholds in the threshold vector 
        train_loss, train_acc, exp_f1, va_avg, au_f1, outputs_new, targets_new = validate(trainloader, model, au_criterion, va_criterion , exp_criterion, epoch, use_cuda, mode='Train Stats')
        
        #Updating the threshold vector using the labels of prediction on the train data
        threshold = adaptive_threshold_generate(outputs_new, targets_new, threshold, epoch)
        
        #Validating the model
        val_loss, val_acc,exp_f1, va_avg, au_f1, _, _ = validate(val_loader, model, au_criterion, va_criterion , exp_criterion,  epoch, use_cuda, mode='Validation Stats')
        
        overall_perf = exp_f1 + va_avg + au_f1
        
        # append logger file
        logger.append([train_loss, train_acc, val_loss, val_acc,exp_f1, va_avg, au_f1, overall_perf])
        
        # save model
        is_best = overall_perf > best_perf
        best_perf = max(overall_perf, best_perf)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': val_acc,
                'best_perf': best_perf,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        val_perf.append(overall_perf)
        
    logger.close()
      
    print('Best Performance:')
    print(best_perf)


def save_checkpoint(state, is_best):
    if is_best:
        torch.save(state, os.path.join(args.model_dir, 'ss-mfar-cp_ep{}.pth.tar'.format(state['epoch'] - 1)))

    
def train(trainloader, model, optimizer, au_criterion, va_criterion , exp_criterion, threshold, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_exp = AverageMeter()
    losses_au = AverageMeter()
    losses_va = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    
    args.train_iteration = int(len(trainloader.dataset)/args.batch_size)

    bar = Bar('Train', max=args.train_iteration)
    labeled_train_iter = iter(trainloader)
    
    model.train()
    loss = {}
    outputs_exp = torch.ones(1).cuda()
    targets_exp = torch.ones(1).long().cuda()
    
    outputs_au =  torch.ones(1, 12).cuda()
    targets_au =  torch.ones(1, 12).long().cuda()
    

    outputs_va =  torch.ones(1, 2).cuda()
    targets_va =  torch.ones(1, 2).float().cuda()
    
    for batch_idx in range(args.train_iteration):
        try:
            (inputs, inputs_strong), label_AU,m_label_AU, label_V,label_A, m_label_VA, label_EXP,m_label_EXP, img_path = labeled_train_iter.next() 
            
        except:
            labeled_train_iter = iter(trainloader)
            (inputs, inputs_strong), label_AU,m_label_AU, label_V,label_A, m_label_VA, label_EXP,m_label_EXP,img_path  = labeled_train_iter.next()

        
        # measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs.size(0)

        if use_cuda:
            inputs, inputs_strong, label_AU, m_label_AU, label_V,label_A,m_label_VA, label_EXP, m_label_EXP = inputs.cuda(), inputs_strong.cuda(), label_AU.cuda(),  m_label_AU.cuda(),label_V.cuda(), label_A.cuda(), m_label_VA.cuda(),label_EXP.cuda(), m_label_EXP.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel(invalid-label) samples
            exp_u_indices = [i for i, v in enumerate(m_label_EXP) if v == 0]
            if len(exp_u_indices) > 0:
                inputs_u = inputs[exp_u_indices]
                outputs_u = model(inputs_u)
                p = torch.softmax(outputs_u['EXPR'], dim=1)
                max_probs, conf_preds = torch.max(p, dim=1)
                conf_preds = conf_preds.detach()
        
        outputs = model(inputs)
        
        # measure accuracy and record loss
        exp_indices = [i for i, v in enumerate(m_label_EXP) if v == True]
        predicted, target = outputs['EXPR'][exp_indices], label_EXP[exp_indices]
        
        #EXPR_Labeled = Entropy loss of proprely labeled samples wrt Expressions
        loss['EXPR_Lab'] = exp_criterion(predicted, target)
        loss['EXPR_Lab'] = loss['EXPR_Lab'].mean()
        
        prec1, prec5 = accuracy(predicted, target, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        outputs_exp = torch.cat((outputs_exp, F.softmax(predicted , dim=-1).argmax(-1).float()), dim=0)
        targets_exp = torch.cat((targets_exp, target), dim=0)
        
        #For unlabelled data using semi-supervision unlabeled data in the current batch
        if len(exp_u_indices) > 0:
            outputs_strong = model(inputs_strong)
            
            mask_u = mask_generate(max_probs, conf_preds, len(conf_preds), threshold)
            mask = mask_generate(max_probs, conf_preds, batch_size, threshold)
            #mask_idx -> is non-confident indices
            non_conf_idx = np.where(mask.cpu() == 0)[0]
            probs_w = F.softmax(outputs['EXPR'], dim=1)
            probs_s = F.softmax(outputs_strong['EXPR'], dim=1)

            #Ls = symmetricKLDiv(non_confident_weak_distr, strong_augmentations_distr)
            #This helps us to learn the features better by making the feature distribution of both weak and strong augmentations similar.
            #EXPR_Con = Consistency loss wrt Expression
            loss['EXPR_Con'] = symmetric_kl_div( probs_w[non_conf_idx], probs_s[non_conf_idx] )
            loss['EXPR_Con'] = loss['EXPR_Con'].mean()
            
            #Lu = CE loss over the confidently predicted weak augmentations with unlabeled data and strong augmentations
            #EXPR_US = unsupervised loss on strong augmentations
            loss['EXPR_US'] = exp_criterion(outputs_strong['EXPR'][exp_u_indices], conf_preds) * mask_u
            loss['EXPR_US'] = loss['EXPR_US'].mean()
            
            #loss = Lx *0.5 + Lu + Ls * 0.1
            loss['EXPR'] = (loss['EXPR_Lab']  + loss['EXPR_US'])*0.5 + 0.5*loss['EXPR_Con']
            
        else:
            loss['EXPR'] = loss['EXPR_Lab']

        au_indices = [i for i, v in enumerate(m_label_AU) if v == True]
        predicted, target = outputs['AU'][au_indices], label_AU[au_indices] 
        loss['AU'] = au_criterion(predicted, target.float())
        outputs_au = torch.cat((outputs_au,  (torch.sigmoid(predicted) > 0.5).float()), dim=0)
        targets_au = torch.cat((targets_au, target), dim=0)
        
        
        va_indices = [i for i, v in enumerate(m_label_VA) if v == True]        
        labels_VA = torch.stack((label_V, label_A), dim=1)
        predicted, target = outputs['VA'][va_indices], labels_VA[va_indices]
        loss['VA'] = 0.5 * ( va_criterion(predicted[:,0],target[:,0] ) + va_criterion(predicted[:,1],target[:,1]) )                         
        outputs_va = torch.cat((outputs_va,  predicted), dim=0)
        targets_va = torch.cat((targets_va, target.float()), dim=0)
        
        # record loss
        losses_exp.update(loss['EXPR'].item(), inputs.size(0))
        losses_au.update(loss['AU'].item(), inputs.size(0))
        losses_va.update(loss['VA'].item(), inputs.size(0))
                        
        #total loss
        totalloss = loss['EXPR'] + loss['VA'] + loss['AU']          
        losses.update(totalloss.item(), inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        totalloss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '[{cur_epoch:}]({batch}/{size})Total:{total:}| TotL:{loss:.4f}| L_exp:{loss_exp:.4f}| L_au:{loss_au:.4f}| L_va:{loss_va:.4f}| Acc:{top1: .4f}'.format(
        			cur_epoch = epoch,
                    batch=batch_idx + 1,
                    size=args.train_iteration,
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    loss_exp = losses_exp.avg,
                    loss_au = losses_au.avg,
                    loss_va = losses_va.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    
    exp_f1 = EXPR_metric(outputs_exp.cpu().numpy() , targets_exp.cpu().numpy())    
    print('exp performance',exp_f1)

    va_avg = VA_metric(outputs_va.cpu().detach().numpy() , targets_va.cpu().numpy())
    print('va performance',va_avg )
    
  
    au_f1 = AU_metric(outputs_au.cpu().numpy() , targets_au.cpu().numpy())    
    print('au performance', au_f1)
    
    print('Overall score ', au_f1 + va_avg + exp_f1  )   
    
    return losses.avg, top1.avg, exp_f1, va_avg, au_f1     
    
    
def validate(dataloader, model,  au_criterion, va_criterion, exp_criterion, epoch, use_cuda, mode):
	'''
	Validate function
	'''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_exp = AverageMeter()
    losses_au = AverageMeter()
    losses_va = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    
    args.validate_iteration = int(len(dataloader.dataset)/args.batch_size)
    print('\n')
    bar = Bar('Validation', max=args.validate_iteration)
    
    labeled_val_iter = iter(dataloader)
    
    model.eval()
    
    loss = {}
    outputs_exp = torch.ones(1).cuda()
    outputs_exp_new = torch.ones(1, args.num_exp_classes).cuda()
    targets_exp = torch.ones(1).long().cuda()
    
    outputs_au =  torch.ones(1, 12).cuda()
    targets_au =  torch.ones(1, 12).long().cuda()
    
    
    outputs_va =  torch.ones(1, 2).cuda()
    targets_va =  torch.ones(1, 2).float().cuda()
            
    with torch.no_grad():
        for batch_idx in range(args.validate_iteration):
            try:
               inputs, label_AU,m_label_AU, label_V,label_A, m_label_VA, label_EXP,m_label_EXP, img_path = labeled_val_iter.next()
               
            except:
               labeled_val_iter = iter(dataloader)
               inputs, label_AU,m_label_AU, label_V,label_A, m_label_VA, label_EXP,m_label_EXP, img_path  = labeled_val_iter.next()
            
            # measure data loading time
            data_time.update(time.time() - end)       
            if mode == 'Train Stats':
                inputs = inputs[0]
                
            batch_size = inputs.size(0)
       
            if use_cuda:
               inputs, label_AU, m_label_AU, label_V,label_A,m_label_VA, label_EXP, m_label_EXP = inputs.cuda(), label_AU.cuda(),  m_label_AU.cuda(),label_V.cuda(), label_A.cuda(), m_label_VA.cuda(),label_EXP.cuda(), m_label_EXP.cuda()
            
               
            outputs = model(inputs)
           
            # measure accuracy and record loss
            exp_indices = [i for i, v in enumerate(m_label_EXP) if v == True]#considering valid annotations
            predicted, target = outputs['EXPR'][exp_indices], label_EXP[exp_indices]
            loss['EXPR'] = exp_criterion(predicted, target)
            loss['EXPR'] = loss['EXPR'].mean()
            prec1, prec5 = accuracy(predicted, target, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            outputs_exp = torch.cat((outputs_exp,  F.softmax(predicted , dim=-1).argmax(-1).float()), dim=0)
            outputs_exp_new = torch.cat((outputs_exp_new, outputs['EXPR'][exp_indices]), dim=0)
            targets_exp = torch.cat((targets_exp, target), dim=0)                
     
            au_indices = [i for i, v in enumerate(m_label_AU) if v == True]#considering valid annotations
            predicted, target = outputs['AU'][au_indices], label_AU[au_indices]
            loss['AU'] = au_criterion(predicted, target.float())
            outputs_au = torch.cat((outputs_au,  (torch.sigmoid(predicted) > 0.5).float()), dim=0)
            targets_au = torch.cat((targets_au, target), dim=0)
           
           
            va_indices = [i for i, v in enumerate(m_label_VA) if v == True]#considering valid annotations     
            labels_VA = torch.stack((label_V, label_A), dim=1)
            predicted, target = outputs['VA'][va_indices], labels_VA[va_indices]
            loss['VA'] = 0.5 * ( va_criterion(predicted[:,0],target[:,0] ) + va_criterion(predicted[:,1],target[:,1]) )                         
            outputs_va = torch.cat((outputs_va,  predicted), dim=0)
            targets_va = torch.cat((targets_va, target.float()), dim=0)
           
            # record loss
            losses_exp.update(loss['EXPR'].item(), inputs.size(0))
            losses_au.update(loss['AU'].item(), inputs.size(0))
            losses_va.update(loss['VA'].item(), inputs.size(0))
                            
            #total loss
            totalloss = loss['EXPR'] + loss['VA'] + loss['AU']          
            losses.update(totalloss.item(), inputs.size(0))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # plot progress
            bar.suffix  = '(({batch}/{size})Total:{total:}| TotL:{loss:.4f}| L_exp:{loss_exp:.4f}| L_au:{loss_au:.4f}| L_va:{loss_va:.4f}| Acc:{top1: .4f}'.format(
                       batch=batch_idx + 1,
                       size=args.validate_iteration,
                       total=bar.elapsed_td,
                       loss=losses.avg,
                       loss_exp = losses_exp.avg,
                       loss_au = losses_au.avg,
                       loss_va = losses_va.avg,
                       top1=top1.avg,
                       )
            bar.next()
    
    bar.finish()
    exp_f1 = EXPR_metric(outputs_exp.cpu().numpy() , targets_exp.cpu().numpy())
    print('exp performance',exp_f1)
    
    va_avg = VA_metric(outputs_va.cpu().detach().numpy() , targets_va.cpu().numpy())
    print('va performance',va_avg )
    
    au_f1 = AU_metric(outputs_au.cpu().numpy() , targets_au.cpu().numpy())  
    print('au performance', au_f1)
    
    print('Overall score ', au_f1 + va_avg + exp_f1 )
    
    return losses.avg, top1.avg, exp_f1, va_avg, au_f1, outputs_exp_new, targets_exp
    

def mask_generate(max_probs, max_idx, batch, threshold):
    """
    0 -> non confident predictions
    1 -> confident predictions
    From Li 2022 AdaCM
    """
    mask_ori = torch.zeros(batch)
    for i in range(args.num_exp_classes):
        idx = np.where(max_idx.cpu() == i)[0]
        max_probs = max_probs.cpu()
        m = max_probs[idx].ge(threshold[i]).float()
        for k in range(len(idx)):
            mask_ori[idx[k]]+=m[k]
    return mask_ori.cuda()

def adaptive_threshold_generate(outputs, targets, threshold, epoch):
    """
    Generates new threshold vector using the latest output predictions
    From Li 2022 AdaCM
    """
    outputs_l = outputs[1:, :]
    targets_l = targets[1:]
    probs = torch.softmax(outputs_l, dim=1)
    max_probs, max_idx = torch.max(probs, dim=1)
    eq_idx = np.where(targets_l.eq(max_idx).cpu() == 1)[0]

    probs_new = max_probs[eq_idx]
    targets_new = targets_l[eq_idx]
    for i in range(args.num_exp_classes):
        idx = np.where(targets_new.cpu() == i)[0]
        if idx.shape[0] != 0:
            threshold[i] = probs_new[idx].mean().cpu() * 0.97 / (1 + math.exp(-1 * epoch)) if probs_new[idx].mean().cpu() * 0.97 / (1 + math.exp(-1 * epoch)) >= 0.8 else 0.8
        else:
            threshold[i] = 0.8
    return threshold
                    
if __name__ == '__main__':
    main()                        

