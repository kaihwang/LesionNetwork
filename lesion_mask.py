import numpy as np
import pandas as pd
import nibabel as nib

masks_path = '/home/kahwang/bkh/Lesions/0.5mm/'

tha_mask = nib.load(masks_path + 'tha_0.5_mask.nii.gz')
tha_size = np.sum(tha_mask.get_fdata()>0)

task_mask = nib.load('/home/kahwang/bkh/Lesions/task_0.5.nii.gz')

list = pd.read_csv('list.txt', header=None, dtype='str')

list = list.rename(columns={0:'subject'})
for s in np.arange(len(list)):
    subject = list.loc[s,'subject']
    mask = nib.load(masks_path + subject +".nii.gz")
    if np.sum(mask.get_fdata()>0) < tha_size:
        list.loc[s, 'include'] = 'Yes'
        list.loc[s, 'size'] = np.sum(mask.get_fdata()>0) * 0.5*0.5*0.5
        list.loc[s, 'task_contrast_mean'] = np.mean(task_mask.get_fdata()[mask.get_fdata()>0])
        list.loc[s, 'task_contrast_sum'] = np.sum(task_mask.get_fdata()[mask.get_fdata()>0])
        if list.loc[s, 'task_contrast_mean'] > 0:
            list.loc[s, 'group'] = 'task positive'
        else:
            list.loc[s, 'group'] = 'task negative'




#
