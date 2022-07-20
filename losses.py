'''

Aum Sri Sai Ram

Implementation of losses used 

Authors: Darshan Gera, Badveeti Naveen Siva Kumar, Bobbili Veerendra Raj Kumar, Dr. S. Balasubramanian, SSSIHL

Date: 20-07-2022

Email: darshangera@sssihl.edu.in

'''

import torch
import torch.nn.functional as F
import torch.nn as nn

categories= {'AU': ['AU1','AU2','AU4','AU6','AU7','AU10','AU12','AU15','AU23','AU24','AU25','AU26'],
            'EXPR': ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise','Other'],
            'VA': ['valence', 'arousal']}
au_names_list = categories['AU']
emotion_names_list = categories['EXPR']


#Weights  used for re-weighting for Expression
class_weights =  [3.780655655655656, 19.900109769484082, 28.612689393939394, 29.034272901985908, 4.998345740281224, 11.912866342489158, 17.338370313695485, 3.6473925639787543]
class_weights = torch.tensor(class_weights)
class_weights = class_weights.float().cuda()

def cross_entropy_loss(y_hat, y):   
    Num_classes = y_hat.size(-1)
    return F.cross_entropy(y_hat, y, weight=class_weights[:Num_classes], reduction='none')
    

#Weights  used for re-weighting for Action-Units
pos_weight = [  2.7605408, 5.90714694,  2.53947498,  1.68789413,  1.24967946,  1.34926605,  1.92876078,  20.99918699,  17.33489933, 10.32540476,  0.73186558,  4.80271476]
pos_weight = torch.tensor(pos_weight)
pos_weight = pos_weight.float().cuda()

def classification_loss_func(y_hat, y):
    loss1 = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight= pos_weight)#, reduction='none')
    return loss1

    
def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    return (p * p.log2() - p * q.log2()).sum(dim=1)

def symmetric_kl_div(p, q):
	#consistency loss
    return kl_div(p, q) + kl_div(q, p)    
   
EPS = 1e-8

class CCCLoss(nn.Module):
	'''
	Used to calculate CCC loss for valence-arousal prediction
	'''
    def __init__(self, digitize_num=20, range=[-1, 1], weight=None):
        super(CCCLoss, self).__init__() 
        self.digitize_num =  digitize_num
        self.range = range
        self.weight = weight
        if self.digitize_num >1:
            bins = np.linspace(*self.range, num= self.digitize_num)
            self.bins = torch.as_tensor(bins, dtype = torch.float32).cuda().view((1, -1))
    def forward(self, x, y): 
        # the target y is continuous value(BS, )
        # the input x is either continuous value(BS, ) or probability output(digitized)
        y = y.view(-1)
        if self.digitize_num !=1:
            x = F.softmax(x, dim=-1)
            x = (self.bins * x).sum(-1) # expectation
        x = x.view(-1)
        if self.weight is None:
            vx = x - torch.mean(x) 
            vy = y - torch.mean(y)
            rho =  torch.sum(vx.double() * vy.double()).float() / (torch.sqrt(torch.sum(torch.pow(vx.float(), 2))) * torch.sqrt(torch.sum(torch.pow(vy.float(), 2))) + EPS)
            x_m = torch.mean(x)
            y_m = torch.mean(y).float()
            x_s = torch.std(x)
            y_s = torch.std(y).float()
            ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2) + EPS)
            ccc = ccc.float()
        else:
            rho = weighted_correlation(x, y, self.weight)
            x_var = weighted_covariance(x, x, self.weight)
            y_var = weighted_covariance(y, y, self.weight)
            x_mean = weighted_mean(x, self.weight)
            y_mean = weighted_mean(y, self.weight)
            ccc = 2*rho*torch.sqrt(x_var)*torch.sqrt(y_var)/(x_var + y_var + torch.pow(x_mean - y_mean, 2) +EPS)
        return (1-ccc).float()
    
