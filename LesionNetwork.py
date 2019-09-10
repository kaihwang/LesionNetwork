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


	#### find control patients that have lesion size similar to thalamic lesion
	sdf = load_corticalPatient_score_size()
	sdf.to_csv('Cortical_Patients.csv')

	yeof = nib.load('/data/backed_up/shared/ROIs/Yeo7network_2mm.nii.gz')
	tha_morel = nib.load('/home/kahwang/ROI_for_share/morel_overlap_2mm.nii.gz')
	mask_sum = np.zeros(yeof.get_data().shape)
	
	# loading mask then sum them for display
	for s in sdf[sdf['Lesion Size']<2536]['Subject'].values:
		fn = '/home/kahwang/Lesion_Masks/%s.nii.gz' %s
		m = nib.load(fn)
		res_m = resample_from_to(m, yeof).get_data()
		mask_sum = mask_sum + res_m

	# plot lesion locations	
	mask_img = nilearn.image.new_img_like(yeof, mask_sum, copy_header=True)
	nilearn.plotting.plot_glass_brain(mask_img)
	nib.save(mask_img, '/home/kahwang/Tha_Lesion_Masks/SizeControlOverlap.nii')



	#do test
	df = pd.read_csv('Neuropsych.csv')
	# compare to control
	scipy.stats.mannwhitneyu(sdf.loc[sdf['Lesion Size']<2536]['Trail Score'].values, df[df['Group']=='Medial']['Trail Making B (seconds)'].values[0:-1])
	# compare between groups
	scipy.stats.mannwhitneyu(df[df['Group']=='Lateral']['Trail Making B (seconds)'].values, df[df['Group']=='Medial']['Trail Making B (seconds)'].values[0:-1])



 	#find all lesions
 	pdf = pd.DataFrame(columns=['Subject', 'Size'])
 	pdf['Subject'] = np.loadtxt('/home/kahwang/Lesion_Masks/list', dtype='str')
 	yeof = nib.load('/data/backed_up/shared/ROIs/Yeo7network_2mm.nii.gz')
 	tha_morel = nib.load('/home/kahwang/ROI_for_share/morel_overlap_2mm.nii.gz').get_data()
 	for i, s in enumerate(pdf['Subject']):
		
		# load mask and get size
		fn = '/home/kahwang/Lesion_Masks/%s.nii.gz' %s
		m = nib.load(fn)
		pdf.loc[i, 'Size'] = np.sum(m.get_data())
		
		#overlap with yeo network
		res_m = resample_from_to(m, yeof).get_data()
		yeo_m = yeof.get_data()

		for ii, n in enumerate(networks):
			networkpartition = yeo_m == ii+1
			overlap = res_m *networkpartition
			pdf.loc[i, str(n) + '_overlap' ] = np.sum(overlap)/ np.float(np.sum(networkpartition))

		#exclude patients with lesions that touches the thalamus	
		if np.sum(res_m*tha_morel)>0:  #if overlap with thalamus
			pdf = pdf.drop(pdf[pdf['Subject']==str(s)].index)			
	
			#biggest thalamus lesion size is 2536 mm3
			# 60 patients 
			# very little correlation between lesion size and performance in these 60 patients
			# in full 600 sample, there is a moderate correlation .18	






