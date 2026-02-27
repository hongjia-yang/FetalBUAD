import numpy as np
import copy
import re
import nibabel as nb
import os
import pandas as pd


metric_path='./metric/'
data_path='./result/'
subject_id=np.load('test.npy')

id_list=[]
value_list=[]
count=0
for ii in range(0,subject_id.shape[0]):
    new_id=subject_id[ii,0]
    std_voxelmap=nb.load(data_path+str(new_id)+'_std_voxelmap.nii.gz').get_fdata()
    mask=nb.load(data_path+str(new_id)+'_mask.nii.gz').get_fdata()
    voxelmap=nb.load(data_path+str(new_id)+'_voxelmap_gap.nii.gz').get_fdata()
    std_max=std_voxelmap[mask>0].max()
    std_min=std_voxelmap[mask>0].min()
    std_voxelmap_norm = (std_voxelmap - std_min) / (std_max - std_min) * (mask>0)+1
    voxelmap=voxelmap*std_voxelmap_norm 
    metrix = np.std(voxelmap[voxelmap>0])
    value_list.append(metrix)
    id_list.append(new_id)
    count=count+1
result_df = pd.DataFrame({'id': id_list,'Anomaly_Score': value_list})
result_df.to_excel(cal_metric_path+'result.xlsx', index=False)