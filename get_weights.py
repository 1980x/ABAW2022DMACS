'''

Aum Sri Sai Ram

Code to obtain the weights used in loss functions for expression classification and action unit detection tasks

Authors: Darshan Gera, Badveeti Naveen Siva Kumar, Bobbili Veerendra Raj Kumar, Dr. S. Balasubramanian, SSSIHL

Date: 20-07-2022

Email: darshangera@sssihl.edu.in

'''

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
filename = '../data/Affwild2/training_set_annotations_22.txt'


exp_dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,-1:0}

au_dict_pos={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,-1:0}
au_dict_neg={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,-1:0}
au_dict_masked={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,-1:0}

def get_image2all(filename):
    with open(filename) as f:
        mtl_lines = f.read().splitlines()      
    print('Total: ', len(mtl_lines))          
    for line in mtl_lines[1:]:
        splitted_line=line.split(',')
        expression=int(splitted_line[3])
        aus=list(map(int,splitted_line[4:]))
        exp_dict[expression]=exp_dict[expression]+1
        
        for i,a in enumerate(aus):
            if a == 1:            
               au_dict_pos[i] = au_dict_pos[i]  + 1               
            elif a == -1:            
               au_dict_masked[i] = au_dict_masked[i] + 1
            else: 
               au_dict_neg[i] = au_dict_neg[i] + 1
    
    s = 0
    for k in range(0,8):
        s = s + exp_dict[k]
    
    exp_wts = []
    for k in range(0,8):
        exp_wts.append(s/exp_dict[k])
    print('Exp weights: ', exp_wts,' total wt: ', sum(exp_wts))
    
    
    num_labels = 12
    aus_class_weights = np.empty([num_labels, 2])
    for i in range(num_labels):
        neg, pos, masked = au_dict_neg[i], au_dict_pos[i], au_dict_masked[i]
        print(neg, pos, masked, neg+ pos+masked)
        total = neg + pos
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
    
        aus_class_weights[i][0]=weight_for_0
        aus_class_weights[i][1]=weight_for_1
        
    print('Au wts: ', aus_class_weights)

get_image2all(filename)       
