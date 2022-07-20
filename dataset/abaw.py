'''

Aum Sri Sai Ram

Implementation of Dataset_ABAW_Affwild2 dataset class

Authors: Darshan Gera, Badveeti Naveen Siva Kumar, Bobbili Veerendra Raj Kumar, Dr. S. Balasubramanian, SSSIHL

Date: 20-07-2022

Email: darshangera@sssihl.edu.in

'''

import cv2
import numpy as np
from PIL import Image
import os
import copy
import csv
import pandas as pd

import torchvision
import torch
import torchvision.transforms.transforms as transforms
import torch.utils.data as data
from .randaugment import RandAugment


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform
        self.strong_transfrom = copy.deepcopy(transform)
        self.strong_transfrom.transforms.insert(0, RandAugment(3,5))

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.strong_transfrom(inp)
        return out1, out2#, out3

def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            img = Image.fromarray(img)
            return img
    except IOError:
        print('Cannot load image ' + path)

class Dataset_ABAW_Affwild2(torch.utils.data.Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader):

        self.root = root
        self.transform = transform
        self.loader = loader

        self.image_list = []
        self.labels_AU =  []
        self.labels_V =  []
        self.labels_A =  []
        self.labels_Exp =  []

        
        with open(file_list) as f:
            img_label_list = f.read().splitlines()[1:]
            
        for info in img_label_list:            
            details = info.split(',')
            if (details[3] == '-1' and details[1] == '-5.0' and details[2] == '-5.0' and '-1' in details[4:]) :
               continue 
              
            self.labels_Exp.append(int(details[3]))     
            self.labels_V.append(float(details[1]))
            self.labels_A.append(float(details[2]))
            self.labels_AU.append(details[4:])
            self.image_list.append(details[0])
            
        print('Total samples: ', len(self.image_list)) 
          
        self.m_labels_Exp = [ x != -1 for x in self.labels_Exp]
        print('Valid Exp samples: ',sum(self.m_labels_Exp)) 
        
        self.m_labels_VA = [ (self.labels_V[i] != -5.0 and  self.labels_A[i] != -5.0) for i in range(len(self.labels_V))]
        print('Valid VA samples: ',sum(self.m_labels_VA))

        self.m_labels_AU = [ '-1' not in self.labels_AU[i]  for i in range(len(self.labels_AU))]
        print('Valid AU samples: ',sum(self.m_labels_AU))
        

    def __getitem__(self, index):

        img_path = self.image_list[index]
        img = self.loader(os.path.join(self.root, img_path))
        
        if self.transform is not None:
            img = self.transform(img)

        label_V = float(self.labels_V[index])
        label_A = float(self.labels_A[index])
        label_exp = int(self.labels_Exp[index])
        
        m_label_exp = self.m_labels_Exp[index]
        m_label_VA = self.m_labels_VA[index]
        m_label_AU = self.m_labels_AU[index]

        label_au = self.labels_AU[index]
        label_au_ = [int(0) for i in range(len(label_au))]
        for i in range(len(label_au)):
            if label_au[i] =='1':
                label_au_[i] = int(1)
            elif label_au[i] =='-1':
                label_au_[i] = int(-1)
        label_au_ = torch.tensor(label_au_)

        return img, label_au_, m_label_AU, label_V, label_A, m_label_VA, label_exp, m_label_exp, img_path

    
    def __len__(self):
        return len(self.image_list)

        
class Dataset_ABAW_test(torch.utils.data.Dataset):

	'''
	This dataset class is used to load the test data
	'''
	def __init__(self, root, file_list, transform=None, loader=img_loader):
		self.root = root
		self.transform = transform
		self.loader = loader
		self.image_list = []

		with open(file_list) as f:
			img_label_list = f.read().splitlines()[1:]
			

		for info in img_label_list:
			self.image_list.append(info)

	def __getitem__(self, index):
		img_path = self.image_list[index]

		img = self.loader(os.path.join(self.root, img_path))

		if self.transform is not None:
			img = self.transform(img)
   
		return img, img_path

	def __len__(self):
		return len(self.image_list)


