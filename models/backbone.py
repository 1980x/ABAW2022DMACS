'''

Aum Sri Sai Ram

Code to the backbone network used along with the task specific classifiers for multi tasks

Authors: Darshan Gera, Badveeti Naveen Siva Kumar, Bobbili Veerendra Raj Kumar, Dr. S. Balasubramanian, SSSIHL

Date: 20-07-2022

Email: darshangera@sssihl.edu.in

'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
        
class AUClassifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, seq_input):
        bs, seq_len = seq_input.size(0), seq_input.size(1)
        weight = self.fc.weight
        bias = self.fc.bias
        seq_input = seq_input.reshape((bs*seq_len, 1, -1)) # bs*seq_len, 1, metric_dim
        weight = weight.unsqueeze(0).repeat((bs, 1, 1))  # bs,seq_len, metric_dim
        weight = weight.view((bs*seq_len, -1)).unsqueeze(-1) #bs*seq_len, metric_dim, 1
        inner_product = torch.bmm(seq_input, weight).squeeze(-1).squeeze(-1) # bs*seq_len
        inner_product = inner_product.view((bs, seq_len))
        return inner_product + bias        

class ResNet_18(nn.Module):
    def __init__(self, num_exp_classes=8, num_au = 12, num_va=2, fc_layer_dim = 16 ):
        super(ResNet_18, self).__init__()
        
        
        ResNet18 = torchvision.models.resnet18(pretrained=False)
        
        checkpoint = torch.load('./models/resnet18_msceleb.pth')
        ResNet18.load_state_dict(checkpoint['state_dict'], strict=True)

        self.base = nn.Sequential(*list(ResNet18.children())[:-2])

        self.output = nn.Sequential(nn.Dropout(0.5), Flatten())
        
        features_dim =  512
        
        self.exp_fc = nn.Sequential(nn.Linear(features_dim, fc_layer_dim), nn.ReLU())
        self.exp_classifier = nn.Linear(fc_layer_dim, num_exp_classes)
        
        self.au_fc = nn.ModuleList([nn.Sequential(nn.Linear(features_dim, fc_layer_dim),
                    nn.ReLU()) for _ in range(num_au)])
                    
        self.au_classifier = AUClassifier(fc_layer_dim, num_au)
        
        self.va_fc = nn.Sequential(nn.Linear(features_dim, fc_layer_dim), nn.ReLU())
        self.va_classifier = nn.Linear(fc_layer_dim, num_va)        

    def forward(self, image):
        outputs = {}
        
        feature_map = self.base(image)
        feature_map = F.avg_pool2d(feature_map, feature_map.size()[2:])        
        feature = self.output(feature_map)
        feature = F.normalize(feature, dim=1)
        
        outputs['EXPR']  =  self.exp_classifier( self.exp_fc(feature) )
        outputs['VA']  =  self.va_classifier( self.va_fc(feature) )
        
        au_feat = []
        for i, au_i_fc in enumerate(self.au_fc):
            au_feat.append(au_i_fc(feature))
        au_feat = torch.stack(au_feat,dim = 1)
        outputs['AU'] =  self.au_classifier(au_feat)
        
        return outputs   
        

if __name__=='__main__':
        model = ResNet_18().cuda()   
        x = torch.rand(2,3,224,224).cuda()
        y = model(x)
        print(y['VA'][:,0].shape,y['VA'][:,1].shape) 
        print(y.keys())
        
        for k,o in y.items():
            print(k, o.size())  

        
