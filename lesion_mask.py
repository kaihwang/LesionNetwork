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
list = list.reset_index()
df = pd.read_csv('/home/kahwang/bkh/Lesions/Comparison_Potential.csv')
tdf = pd.DataFrame()
i=0
for s in np.arange(len(list)):
    subject = list.loc[s,'subject']
    if not np.any(df['subject'].values==int(subject)):
        tdf.loc[i, 'subject'] = subject
        mask = nib.load(masks_path + subject +".nii.gz")
        lesion_mask = mask.get_fdata()>0
        if np.sum(lesion_mask) < tha_size:
            tdf.loc[i, 'smaller_than_thalamus'] = 'Yes'
        else:
            tdf.loc[i, 'smaller_than_thalamus'] = 'No'
        tdf.loc[i, 'size'] = np.sum(lesion_mask) * 0.5*0.5*0.5
        tdf.loc[i, 'task_contrast_mean'] = np.mean(task_mask[lesion_mask])
        tdf.loc[i, 'task_contrast_sum'] = np.sum(task_mask[lesion_mask])
        tdf.loc[i, 'wm_lesion_size'] = np.sum(wm_mask[lesion_mask]) * 0.5*0.5*0.5
        tdf.loc[i, 'wm_percentage'] = tdf.loc[i, 'wm_lesion_size'] / tdf.loc[i, 'size']
        if tdf.loc[i, 'task_contrast_mean'] > 0:
            tdf.loc[i, 'group'] = 'positive'
        else:
            tdf.loc[i, 'group'] = 'negative'
        i = i+1

tdf.loc[(tdf['smaller_than_thalamus']=='Yes') & (tdf['wm_percentage'] < 1)& (tdf['task_contrast_mean']>0), 'control_group'] = 'positive'
tdf.loc[(tdf['smaller_than_thalamus']=='Yes') & (tdf['wm_percentage'] < 1)& (tdf['task_contrast_mean']<0), 'control_group'] = 'negative'
newtdf = tdf[tdf['control_group'].isna() == False]
newtdf = newtdf.drop_duplicates()
df = df.append(newtdf)
df.loc[df['review?']!='N'].to_csv('find_patients.csv')

#df.to_csv('~/RDSS/tmp/find_patients.csv')


#
