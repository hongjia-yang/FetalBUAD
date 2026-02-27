import sys
training_path='../model_training/'
sys.path.append(training_path)
import numpy as np
import torch
from dataset import *
import nibabel as nb
import pandas as pd
from skimage.transform import resize
import torch.nn.functional as F
from sklearn.metrics import r2_score,mean_absolute_error
import matplotlib.pyplot as plt
from net import *
from torch.utils.data import DataLoader
import os
import re
import copy
import SimpleITK as sitk
from sklearn.linear_model import LinearRegression

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
def normalize_image(imgall, imgresall, mask, norm_ch='all'):
    imgall_norm = copy.deepcopy(imgall)
    imgresall_norm = copy.deepcopy(imgresall)
    if norm_ch == 'all':
        norm_ch = np.arange(imgall.shape[-1])
    for jj in norm_ch:
        img = imgall[:, :, :, jj : jj + 1]
        imgres = imgresall[:, :, :, jj : jj + 1]
        img_mean = np.mean(img[mask > 0.5])
        img_std = np.std(img[mask > 0.5])
    return img_mean,img_std
def block_ind(mask, sz_block=64, sz_pad=0):
    # find indices of smallest block that covers whole brain
    tmp = np.nonzero(mask);
    xind = tmp[0]
    yind = tmp[1]
    zind = tmp[2]
    xmin = np.min(xind); xmax = np.max(xind)
    ymin = np.min(yind); ymax = np.max(yind)
    zmin = np.min(zind); zmax = np.max(zind)
    ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax];
    # calculate number of blocks along each dimension
    xlen = xmax - xmin + 1
    ylen = ymax - ymin + 1
    zlen = zmax - zmin + 1
    nx = int(np.ceil(xlen / sz_block)) + sz_pad
    ny = int(np.ceil(ylen / sz_block)) + sz_pad
    nz = int(np.ceil(zlen / sz_block)) + sz_pad
    # determine starting and ending indices of each block
    xstart = xmin
    ystart = ymin
    zstart = zmin
    xend = xmax - sz_block + 1
    yend = ymax - sz_block + 1
    zend = zmax - sz_block + 1
    xind_block = np.round(np.linspace(xstart, xend, nx))
    yind_block = np.round(np.linspace(ystart, yend, ny))
    zind_block = np.round(np.linspace(zstart, zend, nz))
    ind_block = np.zeros([xind_block.shape[0]*yind_block.shape[0]*zind_block.shape[0], 6])
    count = 0
    for ii in np.arange(0, xind_block.shape[0]):
        for jj in np.arange(0, yind_block.shape[0]):
            for kk in np.arange(0, zind_block.shape[0]):
                ind_block[count, :] = np.array([xind_block[ii], xind_block[ii]+sz_block-1, yind_block[jj], yind_block[jj]+sz_block-1, zind_block[kk], zind_block[kk]+sz_block-1])
                count = count + 1
    ind_block = ind_block.astype(int)
    return ind_block, ind_brain
def fill_internal_zeros(mask):
    mask = (mask > 0).astype(np.uint8)
    mask_image = sitk.GetImageFromArray(mask)
    padded_mask = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    padded_image = sitk.GetImageFromArray(padded_mask)
    connected_external = sitk.ConnectedThreshold(padded_image, seedList=[(0, 0, 0)], lower=0, upper=0)
    internal_region = sitk.BinaryNot(connected_external)
    internal_region_np = sitk.GetArrayFromImage(internal_region)[1:-1, 1:-1, 1:-1]
    filled_mask = np.where(internal_region_np > 0, 1, mask)
    return filled_mask
def enable_dropout(model):
    #enable the dropout layers during inference
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()



GA_id=np.load('validation.npy')
subjects_num=GA_id.shape[0]
dpRoot='./reg/'
dpSave='./result/'


torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net= unet().to(device)
model_path = training_path+"checkpoint/MinValLoss.pth"
net.load_state_dict(torch.load(model_path)['net'])
net.eval()
enable_dropout(net)



#run inference on validation set, generate prediction results
ylabel=np.zeros([subjects_num,2])
ypredict=np.zeros([subjects_num,2])
index=0
for ii in range(0,GA_id.shape[0]):
    subject_id=GA_id[ii,1]
    data=nb.load(dpRoot+str(subject_id)+'_reg.nii.gz')
    img_affine=data.affine
    data=data.get_fdata()
    
    data_copy=data.copy()
    data_expand= np.expand_dims(data_copy, -1)
    mask_2 = data_expand > 0
    data_mean,data_std= normalize_image(data_expand,data_expand,mask_2)
    _, ind_brain = block_ind(mask_2,64,0)
    data=extract_brain(data,ind_brain,[128,160,128])

    data[data>0]=(data[data>0]-data_mean)/data_std
    data = data.reshape((1,1,)+data.shape)
    data = torch.tensor(data, dtype=torch.float32)

    ylabel[index,0]=subject_id
    ylabel[index,1]=GA_id[ii,0]
    n_iterations = 36  # MC Dropout steps
    batch_size = 18     
    n_batches = n_iterations // batch_size  
    remainder = n_iterations % batch_size    
    
    all_global_ages = []
    all_voxelmaps = []
    with torch.no_grad():
        for i in range(n_batches):
            data_batch = data.repeat(batch_size, 1, 1, 1, 1)
            globage_batch, _, voxelmap_batch = net(data_batch.to(device))
            all_global_ages.append(globage_batch.cpu().numpy().flatten())
            all_voxelmaps.append(voxelmap_batch.cpu().numpy())
        
        # if remainder > 0:
        #     data_batch = data.repeat(remainder, 1, 1, 1, 1)
        #     globage_batch, _, voxelmap_batch = net(data_batch.to(device))
        #     all_global_ages.append(globage_batch.cpu().numpy().flatten())
        #     all_voxelmaps.append(voxelmap_batch.cpu().numpy())
        
    global_ages = np.concatenate(all_global_ages)  
    voxelmaps = np.concatenate(all_voxelmaps, axis=0)  #
    
    global_mean = np.mean(global_ages)
    global_std = np.std(global_ages)
    mean_voxelmap = np.mean(voxelmaps, axis=0).reshape(128, 160, 128)
    std_voxelmap = np.std(voxelmaps, axis=0).reshape(128, 160, 128)
    
    ypredict[index,0]=global_mean

    voxelmap=mean_voxelmap.copy()
    mask=nb.load(dpRoot+str(subject_id)+'_mask.nii.gz').get_fdata()
    voxelmap=voxelmap*mask
    voxel_age=np.mean(voxelmap[mask>0])
    ypredict[index,1]=voxel_age

    mean_voxelmap=mean_voxelmap*mask
    nb.Nifti1Image(mean_voxelmap,img_affine).to_filename(dpSave+str(subject_id)+'_mean_voxelmap.nii.gz')
    std_voxelmap=std_voxelmap*mask
    nb.Nifti1Image(std_voxelmap,img_affine).to_filename(dpSave+str(subject_id)+'_std_voxelmap.nii.gz')
    index=index+1

np.save('ylabel.npy',ylabel)
np.save('ypredict.npy',ypredict)








#bias correction
ylabel=np.load('ylabel.npy')
ypredict=np.load('ypredict.npy')[:,0]

reg = LinearRegression().fit(ylabel[:,1].reshape(-1, 1), ypredict)
slope = reg.coef_[0]
intercept = reg.intercept_
print(slope,intercept)

corrected_ypredict = (ypredict - intercept) / slope

corr_param=[slope,intercept]
corr_param=np.array(corr_param)
np.save('corr_param.npy',corr_param)

