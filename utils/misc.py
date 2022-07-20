'''

Aum Sri Sai Ram

Some helper functions for PyTorch, including:
    - progress_bar: progress bar mimic xlua.progress.

Authors: Darshan Gera, Badveeti Naveen Siva Kumar, Bobbili Veerendra Raj Kumar, Dr. S. Balasubramanian, SSSIHL

Date: 20-07-2022

Email: darshangera@sssihl.edu.in

'''
import errno
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

__all__ = ['mkdir_p', 'AverageMeter']


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
