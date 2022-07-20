'''

Aum Sri Sai Ram

Code for the metrics used in main

Authors: Darshan Gera, Badveeti Naveen Siva Kumar, Bobbili Veerendra Raj Kumar, Dr. S. Balasubramanian, SSSIHL

Date: 20-07-2022

Email: darshangera@sssihl.edu.in

'''

from __future__ import print_function, absolute_import

import torchvision.transforms as transforms
import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import f1_score
import numpy as np
from scipy.ndimage.filters import gaussian_filter


__all__ = ['accuracy','EXPR_metric', 'VA_metric', 'AU_metric']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    

def averaged_f1_score(input, target):
    N, label_size = input.shape
    
    f1s = []
    for i in range(label_size):
        f1 = f1_score(input[:, i], target[:, i])
        f1s.append(f1)
    return np.mean(f1s), f1s
    
def accuracy_(input, target):
    assert len(input.shape) == 1
    return sum(input==target)/input.shape[0]
    
def averaged_accuracy(x, y):
    assert len(x.shape) == 2
    N, C =x.shape
    accs = []
    for i in range(C):
        acc = accuracy_(x[:, i], y[:, i])
        accs.append(acc)
    return np.mean(accs), accs
    
def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc

def VA_metric(x, y):
    items = [CCC_score(x[:,0], y[:,0]), CCC_score(x[:,1], y[:,1])]
    return sum(items)/len(items)
    
    
def EXPR_metric(x, y): 
    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)
    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)

    f1 = f1_score(x, y, average= 'macro')
    return f1
    
def AU_metric(x, y):
    f1_av, f1s  = averaged_f1_score(x, y)
    x = x.reshape(-1)
    y = y.reshape(-1)
    return f1_av
    
            
def get_metric_func(task):
    if task =='VA':
        return VA_metric
    elif task=='EXPR':
        return EXPR_metric
    elif task=='AU':
        return AU_metric    
