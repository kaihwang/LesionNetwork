import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from nibabel.processing import resample_from_to
import nilearn
import scipy

def cal_tha_lesionFC():
	''' Calcualte average FC for each lesion mask '''
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
	#smf.ols(formula='Trail_B_Score ~ FP ', data=df).fit().summary()
	return df


def cal_corticalPatient_lesion_load_on_thaPatient_lesionNetwork():
	''' calculate thalamic patient's FCmap "load" within eac cortical patient lesion mask'''
	patients = ['0902', 1105, 1692, 1809, 1830, 2092, 2105, 2552, 2697, 2781, 3049, 3184]
	df = pd.read_csv('/home/kahwang/bin/LesionNetwork/Neuropsych.csv')
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

	# Rank mask with fcmap overlap
	for p in patients:
		sdf[str(p)+'_rank'] = sdf[p].rank()


	df.loc[11,'Patient'] = '0902'

	for i, p in enumerate(df['Patient']):
		df.loc[i, 'Cortical_Patietnt_Score'] = sdf.loc[sdf[str(p)+'_rank']>508]['Trail Score'].mean()

	for i, p in enumerate(df['Patient']):
		print(sdf.loc[sdf[str(p)+'_rank']>508]['Subject'])
	df = tha_df
	sdf = cortical_df
	return tha_df, cortical_df


def cal_corticalPatient_lesion_overlap_with_thaPatient_lesionNetwork():
	''' calculate thalamic patient's FCmap overlap with lesion masks from cortical patients'''
	# t value t.266
	### Cortical Patients
	patients = ['0902', 1105, 1692, 1809, 1830, 2092, 2105, 2552, 2697, 2781, 3049, 3184]
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
		df = tha_df
		sdf = cortical_df

	return tha_df, cortical_df



def sort_corticalPatient_overlap_with_corticalFCNetwork():
	'''Sort cortical patients based on network partition'''

	## Yeo labels.
	  # 0            NONE   0   0   0   0
	  # 1     7Networks_1 120  18 134   0 V
	  # 2     7Networks_2  70 130 180   0 SM
	  # 3     7Networks_3   0 118  14   0 DA
	  # 4     7Networks_4 196  58 250   0 CO
	  # 5     7Networks_5 220 248 164   0 LM
	  # 6     7Networks_6 230 148  34   0 FP
	  # 7     7Networks_7 205  62  78   0 DF

	### Cortical Patients
	sdf = pd.DataFrame(columns=['Subject', 'Trail Score'])
	sdf['Subject'] = np.loadtxt('/home/kahwang/Trail_making_part_B_LESYMAP/subList_No_Epilepsy', dtype='str')
	sdf['Trail Score'] =np.loadtxt('/home/kahwang/Trail_making_part_B_LESYMAP/No_Epilepsy/Data/Score.txt')
	networks = ['V', 'SM', 'DA', 'CO', 'Lim', 'FP', 'DF']
	yeof = nib.load('/data/backed_up/shared/ROIs/Yeo7network_2mm.nii.gz')

	for i, s in enumerate(sdf['Subject']):
		fn = '/home/kahwang/Trail_making_part_B_LESYMAP/No_Epilepsy/Masks/%s.nii.gz' %s
		m = nib.load(fn)
		res_m = resample_from_to(m, yeof).get_data()
		yeo_m = yeof.get_data()

		for ii, n in enumerate(networks):

			networkpartition = yeo_m == ii+1
			overlap = res_m *networkpartition
			sdf.loc[i, str(n) + '_overlap' ] = np.sum(overlap)#/np.sum(res_m)

	for ii, n in enumerate(networks):
		sdf[str(n) + '_overlap'+'_rank'] = sdf[str(n) + '_overlap'].rank()

	return sdf


def load_corticalPatient_score_size():
	'''Create cortical patient data frame with score and lesion size'''
	tdf = pd.DataFrame(columns=['Subject', 'Trail Score'])
	tdf['Subject'] = np.loadtxt('/home/kahwang/Trail_making_part_B_LESYMAP/subList_No_Epilepsy', dtype='str')
	tdf['Trail Score'] =np.loadtxt('/home/kahwang/Trail_making_part_B_LESYMAP/No_Epilepsy/Data/Score.txt')

	edf = pd.DataFrame(columns=['Subject', 'Errors'])
	edf['Subject'] = np.loadtxt('/home/kahwang/WCST_LESYMAP/WCST_PE_noThalamus_noEpil/subList', dtype='str')
	edf['Errors'] =np.loadtxt('/home/kahwang/WCST_LESYMAP/WCST_PE_noThalamus_noEpil//scores.txt')

	sdf = pd.merge(tdf, edf, on=['Subject'], how='outer')


	networks = ['V', 'SM', 'DA', 'CO', 'Lim', 'FP', 'DF']
	yeof = nib.load('/data/backed_up/shared/ROIs/Yeo7network_2mm.nii.gz')
	tha_morel = nib.load('/home/kahwang/ROI_for_share/morel_overlap_2mm.nii.gz').get_data() # already same space as yeo template

	for i, s in enumerate(sdf['Subject']):

		# load mask and get size
		fn = '/home/kahwang/Lesion_Masks/%s.nii.gz' %s
		m = nib.load(fn)
		sdf.loc[i, 'Lesion Size'] = np.sum(m.get_data())

		#overlap with yeo network
		res_m = resample_from_to(m, yeof).get_data()
		yeo_m = yeof.get_data()

		for ii, n in enumerate(networks):
			networkpartition = yeo_m == ii+1
			overlap = res_m *networkpartition
			sdf.loc[i, str(n) + '_overlap' ] = np.sum(overlap)/ np.float(np.sum(networkpartition))

		#exclude patients with lesions that touches the thalamus
		if np.sum(res_m*tha_morel)>0:  #if overlap with thalamus
			sdf = sdf.drop(sdf[sdf['Subject']==str(s)].index)

	#exclude cortical thalamus patients
	patients = ['0902', 1105, 1692, 1809, 1830, 2092, 2105, 2552, 2697, 2781, 3049, 3184]
	for p in patients:
		sdf = sdf.drop(sdf[sdf['Subject']==str(p)].index)


	# Do regression of lesion size on score. The r square is .34!
	# results = sm.OLS(sdf['Trail Score'], sm.add_constant(sdf['Lesion Size'])).fit()
	# sdf['Adj Trail Score'] = results.resid
	return sdf


def rank_patient_lesion_based_on_FCNetworkOverlap():
	'''Then sort patients based on network partition'''
	sdf = load_corticalPatient_score_size()
	networks = ['V', 'SM', 'DA', 'CO', 'Lim', 'FP', 'DF']
	yeof = nib.load('/data/backed_up/shared/ROIs/Yeo7network_2mm.nii.gz')

	for i, s in enumerate(sdf['Subject']):
		fn = '/home/kahwang/Trail_making_part_B_LESYMAP/No_Epilepsy/Masks/%s.nii.gz' %s
		m = nib.load(fn)
		res_m = resample_from_to(m, yeof).get_data()
		yeo_m = yeof.get_data()

		for ii, n in enumerate(networks):

			networkpartition = yeo_m == ii+1
			overlap = res_m *networkpartition
			sdf.loc[i, str(n) + '_overlap' ] = np.sum(overlap)/np.sum(res_m)  # sort based on percentage of overlap with the network

	for ii, n in enumerate(networks):
		sdf[str(n) + '_overlap'+'_rank'] = sdf[str(n) + '_overlap'].rank()

	return sdf






if __name__ == "__main__":



	# #do test
	# df = pd.read_csv('Neuropsych.csv')
	# # compare to control
	# scipy.stats.mannwhitneyu(sdf.loc[sdf['Lesion Size']<2536]['Trail Score'].values, df[df['Group']=='Medial']['Trail Making B (seconds)'].values[0:-1])
	# # compare between groups
	# scipy.stats.mannwhitneyu(df[df['Group']=='Lateral']['Trail Making B (seconds)'].values, df[df['Group']=='Medial']['Trail Making B (seconds)'].values[0:-1])



 # 	#find all lesions
 # 	pdf = pd.DataFrame(columns=['Subject', 'Size'])
 # 	pdf['Subject'] = np.loadtxt('/home/kahwang/Lesion_Masks/list', dtype='str')
 # 	yeof = nib.load('/data/backed_up/shared/ROIs/Yeo7network_2mm.nii.gz')
 # 	tha_morel = nib.load('/home/kahwang/ROI_for_share/morel_overlap_2mm.nii.gz').get_data()
 # 	for i, s in enumerate(pdf['Subject']):

	# 	# load mask and get size
	# 	fn = '/home/kahwang/Lesion_Masks/%s.nii.gz' %s
	# 	m = nib.load(fn)
	# 	pdf.loc[i, 'Size'] = np.sum(m.get_data())

	# 	#overlap with yeo network
	# 	res_m = resample_from_to(m, yeof).get_data()
	# 	yeo_m = yeof.get_data()

	# 	for ii, n in enumerate(networks):
	# 		networkpartition = yeo_m == ii+1
	# 		overlap = res_m *networkpartition
	# 		pdf.loc[i, str(n) + '_overlap' ] = np.sum(overlap)/ np.float(np.sum(networkpartition))

	# 	#exclude patients with lesions that touches the thalamus
	# 	if np.sum(res_m*tha_morel)>0:  #if overlap with thalamus
	# 		pdf = pdf.drop(pdf[pdf['Subject']==str(s)].index)

	# 		#biggest thalamus lesion size is 2536 mm3
	# 		# 60 patients
	# 		# very little correlation between lesion size and performance in these 60 patients
	# 		# in full 600 sample, there is a moderate correlation .18


	### find control patients that have overlap with PFC
	df = pd.read_csv('Comparison.csv')

	tha_morel = nib.load('/home/kahwang/ROI_for_share/morel_overlap_2mm.nii.gz')
	MNIPFC = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/Lesion_Masks/MNIPFC.nii.gz')

	# loading mask then sum them for display
	for i, s in enumerate(df['Subject']):

		# load mask and get size
		if s == 802:
			s='0802'

		fn = '/data/backed_up/kahwang/Tha_Neuropsych/Lesion_Masks/%s.nii.gz' %s
		m = nib.load(fn)
		overlap = MNIPFC.get_data() * m.get_data()

		if i == 0:
			mask_sum =  np.zeros(np.shape(m.get_data()))

		if np.sum(overlap) == 0:
			print(s)
			mask_sum = mask_sum + m.get_data()


	# plot lesion locations
	mask_img = nilearn.image.new_img_like(MNIPFC, mask_sum, copy_header=True)
	#nilearn.plotting.plot_glass_brain(mask_img)
	nib.save(mask_img, '/data/backed_up/kahwang/Tha_Neuropsych/Lesion_Masks/SizeControlOverlap.nii')


	#### Acount for lesion size and age
	df = pd.read_csv('Neuropsych.csv')


	## Norm data for TMT from 'Tombaugh et al
	TMTA_norm ={
	'24': {'mean': 22.93, 'sd': 6.87},
	'34': {'mean': 24.40, 'sd': 8.71},
	'44': {'mean': 28.54, 'sd': 10.09},
	'54': {'mean': 31.78, 'sd': 9.93},
	'59+': {'mean': 31.72, 'sd': 10.14},
	'59-': {'mean': 35.10, 'sd': 10.94},
	'64+': {'mean': 31.32, 'sd': 6.96},
	'64-': {'mean': 33.22, 'sd': 9.10},
	'69+': {'mean': 33.84, 'sd': 6.69},
	'69-': {'mean': 39.14, 'sd': 11.84},
	'74+': {'mean': 40.13, 'sd': 14.48},
	'74-': {'mean': 42.47, 'sd': 15.15},
	'79+': {'mean': 41.74, 'sd': 15.32},
	'79-': {'mean': 50.81, 'sd': 17.44},
	'84+': {'mean': 55.32, 'sd': 21.28},
	'84-': {'mean': 58.19, 'sd': 23.31},
	'89+': {'mean': 63.46, 'sd': 29.22},
	'89-': {'mean': 57.56, 'sd': 21.54},
	}

	TMTB_norm ={
	'24': {'mean': 48.97, 'sd': 12.69},
	'34': {'mean': 50.68 , 'sd': 12.36},
	'44': {'mean': 58.46, 'sd': 16.41},
	'54': {'mean': 63.76, 'sd': 14.42},
	'59+': {'mean': 68.74 , 'sd': 21.02},
	'59-': {'mean': 78.84 , 'sd': 19.09},
	'64+': {'mean': 64.58, 'sd': 18.59},
	'64-': {'mean': 74.55, 'sd': 19.55},
	'69+': {'mean': 67.12, 'sd': 9.31},
	'69-': {'mean': 91.32, 'sd': 28.8},
	'74+': {'mean': 86.27, 'sd': 24.07},
	'74-': {'mean': 109.95, 'sd': 35.15},
	'79+': {'mean': 100.68, 'sd': 44.16},
	'79-': {'mean': 130.61, 'sd': 45.74},
	'84+': {'mean': 132.15, 'sd': 42.95},
	'84-': {'mean': 152.74, 'sd': 65.68},
	'89+': {'mean': 140.54, 'sd': 75.38},
	'89-': {'mean': 167.69, 'sd': 78.50},
	}


	for i, s in enumerate(df['Sub']):

		#convert score
		if df.loc[i, 'Age'] <= 34:
			df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['34']['mean']) / TMTA_norm['34']['sd']
		elif 34 < df.loc[i, 'Age'] <= 44:
			df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['44']['mean']) / TMTA_norm['44']['sd']
		elif 44 < df.loc[i, 'Age'] <= 54:
			df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['54']['mean']) / TMTA_norm['54']['sd']

		if (df.loc[i, 'Age'] > 54) & (df.loc[i, 'Educ'] <= 12):
			if 54 < df.loc[i, 'Age'] <= 59:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['59-']['mean']) / TMTA_norm['59-']['sd']
			elif 59 < df.loc[i, 'Age'] <= 64:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['64-']['mean']) / TMTA_norm['64-']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['69-']['mean']) / TMTA_norm['69-']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['69-']['mean']) / TMTA_norm['69-']['sd']
			elif 69 < df.loc[i, 'Age'] <= 74:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['74-']['mean']) / TMTA_norm['74-']['sd']
			elif 74 < df.loc[i, 'Age'] <= 79:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['79-']['mean']) / TMTA_norm['79-']['sd']
			elif 79 < df.loc[i, 'Age'] <= 84:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['84-']['mean']) / TMTA_norm['84-']['sd']
			elif 84 < df.loc[i, 'Age'] <= 89:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['89-']['mean']) / TMTA_norm['89-']['sd']

		if (df.loc[i, 'Age'] > 54) & (df.loc[i, 'Educ'] > 12):
			if 54 < df.loc[i, 'Age'] <= 59:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['59+']['mean']) / TMTA_norm['59+']['sd']
			elif 59 < df.loc[i, 'Age'] <= 64:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['64+']['mean']) / TMTA_norm['64+']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['69+']['mean']) / TMTA_norm['69+']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['69+']['mean']) / TMTA_norm['69+']['sd']
			elif 69 < df.loc[i, 'Age'] <= 74:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['74+']['mean']) / TMTA_norm['74+']['sd']
			elif 74 < df.loc[i, 'Age'] <= 79:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['79+']['mean']) / TMTA_norm['79+']['sd']
			elif 79 < df.loc[i, 'Age'] <= 84:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['84+']['mean']) / TMTA_norm['84+']['sd']
			elif 84 < df.loc[i, 'Age'] <= 89:
				df.loc[i, 'TMTA_z'] = (df.loc[i, 'TMTA'] - TMTA_norm['89+']['mean']) / TMTA_norm['89+']['sd']


		if df.loc[i, 'Age'] <= 34:
			df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['34']['mean']) / TMTB_norm['34']['sd']
		elif 34 < df.loc[i, 'Age'] <= 44:
			df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['44']['mean']) / TMTB_norm['44']['sd']
		elif 44 < df.loc[i, 'Age'] <= 54:
			df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['54']['mean']) / TMTB_norm['54']['sd']

		if (df.loc[i, 'Age'] > 54) & (df.loc[i, 'Educ'] > 12):
			if 54 < df.loc[i, 'Age'] <= 59:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['59+']['mean']) / TMTB_norm['59+']['sd']
			elif 59 < df.loc[i, 'Age'] <= 64:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['64+']['mean']) / TMTB_norm['64+']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['69+']['mean']) / TMTB_norm['69+']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['69+']['mean']) / TMTB_norm['69+']['sd']
			elif 69 < df.loc[i, 'Age'] <= 74:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['74+']['mean']) / TMTB_norm['74+']['sd']
			elif 74 < df.loc[i, 'Age'] <= 79:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['79+']['mean']) / TMTB_norm['79+']['sd']
			elif 79 < df.loc[i, 'Age'] <= 84:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['84+']['mean']) / TMTB_norm['84+']['sd']
			elif 84 < df.loc[i, 'Age'] <= 89:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['89+']['mean']) / TMTB_norm['89+']['sd']

		if (df.loc[i, 'Age'] > 54) & (df.loc[i, 'Educ'] <= 12):
			if 54 < df.loc[i, 'Age'] <= 59:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['59-']['mean']) / TMTB_norm['59-']['sd']
			elif 59 < df.loc[i, 'Age'] <= 64:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['64-']['mean']) / TMTB_norm['64-']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['69-']['mean']) / TMTB_norm['69-']['sd']
			elif 64 < df.loc[i, 'Age'] <= 69:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['69-']['mean']) / TMTB_norm['69-']['sd']
			elif 69 < df.loc[i, 'Age'] <= 74:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['74-']['mean']) / TMTB_norm['74-']['sd']
			elif 74 < df.loc[i, 'Age'] <= 79:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['79-']['mean']) / TMTB_norm['79-']['sd']
			elif 79 < df.loc[i, 'Age'] <= 84:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['84-']['mean']) / TMTB_norm['84-']['sd']
			elif 84 < df.loc[i, 'Age'] <= 89:
				df.loc[i, 'TMTB_z'] = (df.loc[i, 'TMTB'] - TMTB_norm['89-']['mean']) / TMTB_norm['89-']['sd']

		# load mask and get size
		if s == '902':
			s = '0902'
		if s == '802':
			s =='0802'

		fn = '/data/backed_up/kahwang/Tha_Neuropsych/Lesion_Masks/%s.nii.gz' %s
		try:
			m = nib.load(fn)
			df.loc[i, 'Lesion_Size'] = np.sum(m.get_data())
		except:
			continue

	df['TMTB-A_z'] = df['TMTB_z'] - df['TMTA_z']

	model = smf.ols(formula='TMTB-A_z ~ Lesion_Size', data=df).fit()
	model.summary()
	for i, s in enumerate(df['Sub']):
		df.loc[i, 'TMTB-A_adj'] = df.loc[i, 'TMTB-A_z'] - ((model.params['Lesion_Size']* df.loc[i, 'Lesion_Size']) +model.params['Intercept'])


	model = smf.ols(formula='TMTB_z ~ Lesion_Size', data=df).fit()
	model.summary()

	for i, s in enumerate(df['Sub']):
		df.loc[i, 'TMTB_adj'] = df.loc[i, 'TMTB_z'] - ((model.params['Lesion_Size']* df.loc[i, 'Lesion_Size']) +model.params['Intercept'])

	model = smf.ols(formula='TMTA_z ~ Lesion_Size', data=df).fit()
	model.summary()

	for i, s in enumerate(df['Sub']):
		df.loc[i, 'TMTA_adj'] = df.loc[i, 'TMTA_z'] - ((model.params['Lesion_Size']* df.loc[i, 'Lesion_Size']) +model.params['Intercept'])

	model = smf.ols(formula='PE ~ Lesion_Size + Age + Educ', data=df).fit()
	model.summary()

	for i, s in enumerate(df['Sub']):
		df.loc[i, 'PE_adj'] = df.loc[i, 'PE'] - ((model.params['Lesion_Size']* df.loc[i, 'Lesion_Size']) +(model.params['Educ']* df.loc[i, 'Educ']) +(model.params['Age']* df.loc[i, 'Age']) + model.params['Intercept'])

	model = smf.ols(formula='Boston ~ Lesion_Size + Age + Educ', data=df).fit()
	model.summary()

	for i, s in enumerate(df['Sub']):
		df.loc[i, 'Boston_adj'] = df.loc[i, 'Boston'] - ((model.params['Lesion_Size']* df.loc[i, 'Lesion_Size']) +(model.params['Educ']* df.loc[i, 'Educ']) +(model.params['Age']* df.loc[i, 'Age']) + model.params['Intercept'])


	df.to_csv('Neuropsych_adj.csv')
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['TMTB_adj'].values, df.loc[df['Site']=='Th']['TMTB_adj'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['TMTA_adj'].values, df.loc[df['Site']=='Th']['TMTA_adj'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['TMTB_z'].values, df.loc[df['Site']=='Th']['TMTB_z'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['TMTA_z'].values, df.loc[df['Site']=='Th']['TMTA_z'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['TMTB-A_z'].values, df.loc[df['Site']=='Th']['TMTB-A_z'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['TMTB-TMTA'].values, df.loc[df['Site']=='Th']['TMTB-TMTA'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['TMTB-A_adj'].values, df.loc[df['Site']=='Th']['TMTB-A_adj'].values)

	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['PE_adj'].values, df.loc[df['Site']=='Th']['PE_adj'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Group']=='Medial Thalamus']['TMTB_adj'].values, df.loc[df['Group']=='Lateral Thalamus']['TMTB_adj'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Group']=='Medial Thalamus']['TMTB-TMTA'].values, df.loc[df['Group']=='Lateral Thalamus']['TMTB-TMTA'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Group']=='Medial Thalamus']['TMTB-A_z'].values, df.loc[df['Group']=='Lateral Thalamus']['TMTB-A_z'].values)

	#WCST tests
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['PE'].values, df.loc[df['Site']=='Th']['PE'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['Correct'].values, df.loc[df['Site']=='Th']['Correct'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['Error'].values, df.loc[df['Site']=='Th']['Error'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['PE_response'].values, df.loc[df['Site']=='Th']['PE_response'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['Catogories'].values, df.loc[df['Site']=='Th']['Catogories'].values)

	#Control
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['Boston'].values, df.loc[df['Site']=='Th']['Boston'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['RAVL_Hit'].values, df.loc[df['Site']=='Th']['RAVL_Hit'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['RAVL_Rejection'].values, df.loc[df['Site']=='Th']['RAVL_Rejection'].values)

	#IQs
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['VIQ'].values, df.loc[df['Site']=='Th']['VIQ'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['PIQ'].values, df.loc[df['Site']=='Th']['PIQ'].values)
	scipy.stats.mannwhitneyu(df.loc[df['Site']=='ctx']['FSIQ'].values, df.loc[df['Site']=='Th']['FSIQ'].values)


	### Get the lesion size within AN, MD, Pu and other major nuclei

	#2mm lesion mask /data/backed_up/kahwang/Tha_Neuropsych/Lesion_Masks/2mm/rsmask_CA001.nii.gz
	Morel = {
	1: 'AN',
	2: 'VM',
	3: 'VL',
	4: 'MGN',
	5: 'MD',
	6: 'PuA',
	7: 'LP',
	8: 'IL',
	9: 'VA',
	10: 'Po',
	11: 'LGN',
	12: 'PuM',
	13: 'PuI',
	14: 'PuL',
	17: 'VP'
	}

	for s in df.loc[df['Site']=='Th']['Sub']:

		if s == '902':
			s = '0902'

		fn = '/data/backed_up/kahwang/Tha_Neuropsych/Lesion_Masks/rsmask_%s.nii.gz' %s
		m = nib.load(fn).get_data()
		morel_atlas = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_data()

		for n in [1, 5, 12, 3, 17]:
			if s == '0902':
				s = '902'
			df.loc[df['Sub']==s, Morel[n]] = 8 * np.sum(m[morel_atlas==n])

	#Contrast AN and Pu nuclei
	scipy.stats.mannwhitneyu(df.loc[df['AN']>0]['TMTB_z'].values, df.loc[df['AN']==0]['TMTB_z'].values)
	scipy.stats.mannwhitneyu(df.loc[df['AN']>0]['TMTB-A_z'].values, df.loc[df['AN']==0]['TMTB-A_z'].values)
	scipy.stats.mannwhitneyu(df.loc[df['PuM']>0]['TMTB_z'].values, df.loc[df['PuM']==0]['TMTB_z'].values)
	scipy.stats.mannwhitneyu(df.loc[df['PuM']>0]['TMTB-A_z'].values, df.loc[df['PuM']==0]['TMTB-A_z'].values)
