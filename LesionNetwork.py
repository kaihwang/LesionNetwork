import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from nibabel.processing import resample_from_to

## Yeo labels.
  # 0            NONE   0   0   0   0
  # 1     7Networks_1 120  18 134   0 V
  # 2     7Networks_2  70 130 180   0 SM
  # 3     7Networks_3   0 118  14   0 DA
  # 4     7Networks_4 196  58 250   0 CO
  # 5     7Networks_5 220 248 164   0 LM
  # 6     7Networks_6 230 148  34   0 FP
  # 7     7Networks_7 205  62  78   0 DF


## Calcualte average FC for each lesion mask
patients = ['0902', 1105, 1692, 1809, 1830, 2092, 2105, 2552, 2697, 2781, 3049, 3184]
df = pd.read_csv('/home/kahwang/bin/LesionNetwork/Neuropsych.csv')
yeonetwork = nib.load('/data/backed_up/shared/ROIs/Yeo7network_2mm.nii.gz').get_data()
networks = ['V', 'SM', 'DA', 'CO', 'Lim', 'FP', 'DF']

for p in patients:
	fcfile = '/home/kahwang/bsh/Tha_Lesion_Mapping/NKI_ttest_%s.nii.gz'  %p
	fcmap = nib.load(fcfile).get_data()[:,:,:,0,0]
	print(np.mean(fcmap[yeonetwork==6]))
	
	if p == '0902':
		p=902

	for i, n in enumerate(networks):
		df.loc[df['Patient']==p, n] =  np.mean(fcmap[yeonetwork==i+1])

for n in networks:
	print(np.corrcoef(df[n], df['Trail Making B (seconds)']))


#p=sns.regplot(x='Trail Making B (seconds)', y = 'SM', data=df)

df['Trail_B_Score'] = df['Trail Making B (seconds)']
smf.ols(formula='Trail_B_Score ~ FP ', data=df).fit().summary()


### Cortical Patients
sdf = pd.DataFrame(columns=['Subject', 'Trail Score'])
sdf['Subject'] = np.loadtxt('/home/kahwang/Trail_making_part_B_LESYMAP/subList_No_Epilepsy', dtype='str')
sdf['Trail Score'] =np.loadtxt('/home/kahwang/Trail_making_part_B_LESYMAP/No_Epilepsy/Data/Score.txt')

yeof = nib.load('/data/backed_up/shared/ROIs/Yeo7network_2mm.nii.gz')

for i, s in enumerate(sdf['Subject']):
	fn = '/home/kahwang/Trail_making_part_B_LESYMAP/No_Epilepsy/Masks/%s.nii.gz' %s
	m = nib.load(fn)
	res_m = resample_from_to(m, yeof)
	masked_m = yeof.get_data() * res_m.get_data()

	for p in patients:
		fcfile = '/home/kahwang/bsh/Tha_Lesion_Mapping/NKI_ttest_%s.nii.gz'  %p
		fcmap = nib.load(fcfile).get_data()[:,:,:,0,0]
		sdf.loc[i, p] = np.sum(fcmap * masked_m)

# Rank Cortical Patients
for p in patients:
	sdf[str(p)+'_rank'] = sdf[p].rank()


df.loc[11,'Patient'] = '0902'

for i, p in enumerate(df['Patient']):
	df.loc[i, 'Cortical_Patietnt_Score'] = sdf.loc[sdf[str(p)+'_rank']>508]['Trail Score'].mean()

for i, p in enumerate(df['Patient']):
	print(sdf.loc[sdf[str(p)+'_rank']>508]['Subject'])



# Do spatial mask vin diagram calculation for cortical patients
# t value t.266

### Cortical Patients
sdf = pd.DataFrame(columns=['Subject', 'Trail Score'])
sdf['Subject'] = np.loadtxt('/home/kahwang/Trail_making_part_B_LESYMAP/subList_No_Epilepsy', dtype='str')
sdf['Trail Score'] =np.loadtxt('/home/kahwang/Trail_making_part_B_LESYMAP/No_Epilepsy/Data/Score.txt')

#yeof = nib.load('/data/backed_up/shared/ROIs/Yeo7network_2mm.nii.gz')

for i, s in enumerate(sdf['Subject']):
	fn = '/home/kahwang/Trail_making_part_B_LESYMAP/No_Epilepsy/Masks/%s.nii.gz' %s
	m = nib.load(fn)
	res_m = resample_from_to(m, yeof).get_data()
	yeo_m = yeof.get_data()>=1 
	masked_m = yeo_m * res_m


	for p in patients:
		fcfile = '/home/kahwang/bsh/Tha_Lesion_Mapping/NKI_ttest_%s.nii.gz'  %p
		fcmap = nib.load(fcfile).get_data()[:,:,:,0,1]
		fcmap = fcmap>3.66 
		overlap = fcmap *masked_m
		sdf.loc[i, str(p) + '_overlap' ] = np.sum(overlap)/np.sum(res_m) 

# Rank Cortical Patients
for p in patients:
	sdf[str(p) + '_overlap'+'_rank'] = sdf[str(p) + '_overlap'].rank()


df.loc[11,'Patient'] = '0902'

for i, p in enumerate(df['Patient']):
	df.loc[i, 'Cortical_Patietnt_Score'] = sdf.loc[sdf[str(p) + '_overlap'+'_rank']>518]['Trail Score'].mean()

for i, p in enumerate(df['Patient']):
	print(sdf.loc[sdf[str(p)+'_rank']>508]['Subject'])




### Sort cortical patients based on network partition






