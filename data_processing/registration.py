import os
import nibabel as nb
import pandas as pd
import copy
import numpy as np
import ants

df_numpy=np.load('subject.npy')
print(df_numpy.shape)


def register(patient_file,template_file,new_file):
    # Load the template and patient files
    template_image = ants.image_read(template_file)
    patient_image = ants.image_read(patient_file)
    # Perform registration using ANTS
    registration = ants.registration(fixed=template_image, moving=patient_image, type_of_transform='Affine')
    reg_img = registration['warpedmovout']  
    # save the registered roi
    ants.image_write(reg_img, new_file)


dpRoot='./origin_data/'
dpReg='./registration/'

for index in range(0,df_numpy.shape[0]):
    subject_id=df_numpy[index,0]
    register(dpRoot+str(subject_id)+'/origin_brain.nii.gz','./atlas/STA38.nii.gz',dpReg+str(subject_id)+'_reg.nii.gz')
    data=nb.load(dpReg+str(subject_id)+'_reg.nii.gz')
    img_affine=data.affine
    data=data.get_fdata()
    if np.min(data)<0:
        data[data<0]=0
        nb.Nifti1Image(data,img_affine).to_filename(dpReg+str(subject_id)+'_reg.nii.gz')