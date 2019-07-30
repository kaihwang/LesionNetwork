import numpy as np
import pandas as pd
import nibabel as nib

# Yeo labels.
  # 0            NONE   0   0   0   0
  # 1     7Networks_1 120  18 134   0 V
  # 2     7Networks_2  70 130 180   0 SM
  # 3     7Networks_3   0 118  14   0 DA
  # 4     7Networks_4 196  58 250   0 CO
  # 5     7Networks_5 220 248 164   0 LM
  # 6     7Networks_6 230 148  34   0 FP
  # 7     7Networks_7 205  62  78   0 DF


patients = ['0902', 1105, 1692, 1809, 1830, 2092, 2105, 2552, 2697, 2781, 3049, 3184]
df = pd.read_csv('/home/kahwang/bin/LesionNetwork/Neuropsych.csv')
yeonetwork = nib.load('/data/backed_up/shared/ROIs/Yeo7network_2mm.nii.gz').get_data()

for p in patients:
	fcfile = '/home/kahwang/bsh/Tha_Lesion_Mapping/NKI_ttest_%s.nii.gz'  %p
	fcmap = nib.load(fcfile).get_data()[:,:,:,0,0]
	print(np.mean(fcmap[yeonetwork==6]))
	
	if p == '0902':
		p=902

	df.loc[df['Patient']==p, 'FPfc'] =  np.mean(fcmap[yeonetwork==6])

