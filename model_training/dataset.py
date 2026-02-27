import numpy as np
import torch
import nibabel as nb
import pandas as pd
import random
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter
import math

def extract_brain(data, inds, sz_brain):
    if isinstance(sz_brain, int):
        sz_brain = [sz_brain, sz_brain, sz_brain]
    xsz_brain = inds[1] - inds[0] + 1
    ysz_brain = inds[3] - inds[2] + 1
    zsz_brain = inds[5] - inds[4] + 1
    brain = np.zeros((sz_brain[0], sz_brain[1], sz_brain[2]))
    x_start = int((sz_brain[0] - xsz_brain) / 2)
    y_start = int((sz_brain[1] - ysz_brain) / 2)
    z_start = int((sz_brain[2] - zsz_brain) / 2)
    brain[x_start:x_start+xsz_brain,y_start:y_start+ysz_brain,
          z_start:z_start+zsz_brain] = data[inds[0]:inds[1]+1,inds[2]:inds[3]+1,inds[4]:inds[5]+1]
    return brain
def normalize_image(img, mask):
    img_mean = np.mean(img[mask])
    img_std = np.std(img[mask])
    return img_mean,img_std
def random_flip_3d(image):
    axis = 3  # 默认值，表示没有翻转
    if random.random() > 0.5:
        axis = random.choice([0,1])
        #axis=0
        image = np.flip(image, axis).copy()
    return image, axis

class FetalBrainAgeDataset(torch.utils.data.Dataset):
    def __init__(self, setname):
        self.setname = setname
        if setname=='train':
          t1_data=pd.read_json("train.json")
        elif setname=='val':
          t1_data=pd.read_json("val.json")
        self.t1_data = t1_data

    def __len__(self):
        return len(self.t1_data)
    
    def __getitem__(self, index):
        #preprocessing for T2w brain volume
        X_T1 = nb.load((self.t1_data.iat[index,1])).get_fdata()
        #crop background
        ind_brain = [self.t1_data.iat[index,4], self.t1_data.iat[index,5], self.t1_data.iat[index,6], self.t1_data.iat[index,7], self.t1_data.iat[index,8], self.t1_data.iat[index,9]]
        X_T1 = extract_brain(X_T1, ind_brain, [128, 160, 128])
        #data augmentation
        X_T1, axis = random_flip_3d(X_T1)
        t2_mask=X_T1>0
        t2_mask = t2_mask.reshape((1,) + t2_mask.shape)
        t2_mask = torch.tensor(t2_mask, dtype=torch.float32)
        #normalization
        high_mean,high_std= normalize_image(X_T1,t2_mask)
        X_T1[t2_mask] = (X_T1[t2_mask] - high_mean)/high_std
        X_T1=X_T1*t2_mask
        X_T1 = X_T1.reshape((1,) + X_T1.shape)
        data = torch.tensor(X_T1, dtype=torch.float32)

        #preprocessing for segmentation label
        X_seg = nb.load((self.t1_data.iat[index,1]).replace('_reg','_reg_seg')).get_fdata()
        X_seg = extract_brain(X_seg, ind_brain, [128, 160, 128])
        if axis != 3:
            X_seg = np.flip(X_seg, axis).copy()
        X_seg = X_seg.astype(np.int32)
        seg = torch.tensor(X_seg, dtype=torch.float32)  

        #clinical GA
        label = self.t1_data.iat[index,10]
        label = label.reshape((1,) + label.shape)
        y = torch.tensor(label, dtype=torch.float32)


        return {"image": data, "label": y, "seg": seg, "t2_mask":t2_mask}
