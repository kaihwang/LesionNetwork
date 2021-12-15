import numpy as np
import pandas as pd
import nibabel as nib

masks_path = '/home/kahwang/bkh/Lesions/0.5mm/'

tha_mask = nib.load(masks_path + 'tha_0.5_mask.nii.gz')
tha_size = np.sum(tha_mask.get_fdata()>0) * 0.5*0.5*0.5
wm_mask = nib.load(masks_path + 'WM_mask.nii.gz').get_fdata()
task_mask = nib.load('/home/kahwang/bkh/Lesions/task_0.5.nii.gz').get_fdata()

list = pd.read_csv(masks_path+'list.txt', header=None, dtype='str')
list = list.drop_duplicates()

list = list.rename(columns={0:'subject'})
for s in np.arange(len(list)):
    subject = list.loc[s,'subject']
    mask = nib.load(masks_path + subject +".nii.gz")
    lesion_mask = mask.get_fdata()>0
    if np.sum(lesion_mask) < tha_size:
        list.loc[s, 'smaller_than_thalamus'] = 'Yes'
    else:
        list.loc[s, 'smaller_than_thalamus'] = 'No'
    list.loc[s, 'size'] = np.sum(lesion_mask) * 0.5*0.5*0.5
    list.loc[s, 'task_contrast_mean'] = np.mean(task_mask[lesion_mask])
    list.loc[s, 'task_contrast_sum'] = np.sum(task_mask[lesion_mask])
    list.loc[s, 'wm_lesion_size'] = np.sum(wm_mask[lesion_mask]) * 0.5*0.5*0.5
    list.loc[s, 'wm_percentage'] = list.loc[s, 'wm_lesion_size'] / list.loc[s, 'size']
    if list.loc[s, 'task_contrast_mean'] > 0:
        list.loc[s, 'group'] = 'positive'
    else:
        list.loc[s, 'group'] = 'negative'

list.loc[(list['smaller_than_thalamus']=='Yes') & (list['wm_percentage'] < .4)& (list['task_contrast_mean']>0), 'control_group'] = 'positive'
list.loc[(list['half_thalamus']=='Yes') & (list['wm_percentage'] < .4)& (list['task_contrast_mean']<0), 'control_group'] = 'negative'
newList = list[list['control_group'].isna() == False]
newList = newList.drop_duplicates()
newList.to_csv('~/RDSS/tmp/find_patients.csv')


#
