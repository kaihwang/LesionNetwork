from __future__ import division
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from nibabel.processing import resample_from_to
import nilearn
import scipy 
import matplotlib.pyplot as plt
plt.ion()




if __name__ == "__main__":


	#path to Yeo cortical network parcellation files
	#/home/kahwang/bsh/Corticothalamic_Network/Yeo_Networks_Analyses/Yeo_Networks/Yeo_Binary_All_Networks_Combined/Cortex/
	#Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask_MNI152_1mm.nii.gz
	#Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_MNI152_1mm.nii.gz
	#Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask_MNI152_1mm.nii.gz
	#Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_MNI152_1mm.nii.gz

	#path to thalamus
	#/home/kahwang/bsh/Corticothalamic_Network/Yeo_Networks_Analyses/Yeo_Networks/ThalamusParcellation
	#1000subjects_TightThalamus_clusters007_ref_1mm.nii.gz
	#1000subjects_TightThalamus_clusters007_ref.nii.gz
	#1000subjects_TightThalamus_clusters017_ref_1mm.nii.gz
	#1000subjects_TightThalamus_clusters017_ref.nii.gz
 	
 	#LESY map results
 	#/home/kahwang/bsh/Corticothalamic_Network/Yeo_Networks_Analyses/Lesion_Mapping/Multivariate_Lesion_Mapping/Yeo_17_Net_01
 	#look for stat_img.nii.gz

 	#Buckner98 FC lesion network map results
 	#/home/kahwang/bsh/Corticothalamic_Network/Yeo_Networks_Analyses/Network_Mapping/LESYMAP_zmaps/1mm
 	#zmap_LESYMAP_Yeo_17_Net_17_thr_1mm.nii.gz

 	#network names
 	#https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal

 	### check lesy map output overlap with yeo network 17
	fn = '/home/kahwang/bsh/Corticothalamic_Network/Yeo_Networks_Analyses/Yeo_Networks/Yeo_Binary_All_Networks_Combined/Cortex/' + 'Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_MNI152_1mm.nii.gz'
	yeo_mask = nib.load(fn).get_data()

	#check mask
	for i in np.arange(17):
		print(np.sum(yeo_mask==i+1))


	#Check simulated score lesymap results overlap with Yeo's parcellation
	parcels = ['01' ,'02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17']

	df = pd.DataFrame(columns=['Simulated Score for network #', 'Overlap with Yeo network #', 'voxels count', 'percent overlap'])

	r = 0 
	for p in parcels:
		fn = '/home/kahwang/bsh/Corticothalamic_Network/Yeo_Networks_Analyses/Lesion_Mapping/Multivariate_Lesion_Mapping/Yeo_17_Net_%s/stat_img.nii.gz' %p
		lesion_load = nib.load(fn).get_data()
		# threshold at .5 accoring to Aaron
		lesion_load[lesion_load <.5] = 0

		#check overlap
		for i in np.arange(17):

			df.loc[r, 'Simulated Score for network #'] = p
			df.loc[r, 'Overlap with Yeo network #'] = i+1
			df.loc[r, 'voxels count'] = np.sum(lesion_load[yeo_mask==i+1]>0)
			df.loc[r, 'percent overlap'] = np.sum(lesion_load[yeo_mask==i+1]>0) / np.sum(yeo_mask==i+1)
			r=r+1


	sns.catplot(x='Simulated Score for network #', y='percent overlap', hue='Overlap with Yeo network #', data=df, height=6, kind="bar", palette="muted")		



	### check lesy map output overlap with yeo 7 network partitions
	fn = '/home/kahwang/bsh/Corticothalamic_Network/Yeo_Networks_Analyses/Yeo_Networks/Yeo_Binary_All_Networks_Combined/Cortex/' + 'Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_MNI152_1mm.nii.gz'
	yeo_mask = nib.load(fn).get_data()

	#check mask
	for i in np.arange(7):
		print(np.sum(yeo_mask==i+1))

	#Check simulated score lesymap results overlap with Yeo's parcellation
	parcels = ['01' ,'02', '03', '04', '05', '06', '07']

	df = pd.DataFrame(columns=['Simulated Score for network #', 'Overlap with Yeo network #', 'voxels count', 'percent overlap'])

	r = 0 
	for p in parcels:
		fn = '/home/kahwang/bsh/Corticothalamic_Network/Yeo_Networks_Analyses/Lesion_Mapping/Multivariate_Lesion_Mapping/Yeo_7_Net_%s/stat_img.nii.gz' %p
		lesion_load = nib.load(fn).get_data()
		# threshold at .5 accoring to Aaron
		lesion_load[lesion_load <.5] = 0

		#check overlap
		for i in np.arange(7):

			df.loc[r, 'Simulated Score for network #'] = p
			df.loc[r, 'Overlap with Yeo network #'] = i+1
			df.loc[r, 'voxels count'] = np.sum(lesion_load[yeo_mask==i+1]>0)
			df.loc[r, 'percent overlap'] = np.sum(lesion_load[yeo_mask==i+1]>0) / np.sum(yeo_mask==i+1)
			r=r+1


	sns.catplot(x='Simulated Score for network #', y='percent overlap', hue='Overlap with Yeo network #', data=df, height=6, kind="bar", palette="muted")		




	### check overlap with yeo thalamus 17 network parcellation
	fn = '/home/kahwang/bsh/Corticothalamic_Network/Yeo_Networks_Analyses/Yeo_Networks/ThalamusParcellation/' + '1000subjects_TightThalamus_clusters017_ref_1mm.nii.gz'
	yeo_mask = nib.load(fn).get_data()

	#check mask
	for i in np.arange(17):
		print(np.sum(yeo_mask==i+1))

	#Check simulated score lesymap results overlap with Yeo's parcellation
	parcels = ['01' ,'02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17']

	df = pd.DataFrame(columns=['Lesymap seed network #', 'Overlap with Yeo thalamus parcel #', 'zscore'])
	r = 0 
	for p in parcels:
		fn = '/home/kahwang/bsh/Corticothalamic_Network/Yeo_Networks_Analyses/Network_Mapping/LESYMAP_zmaps/1mm/zmap_LESYMAP_Yeo_17_Net_%s_thr_1mm.nii.gz' %p
		zmap = nib.load(fn).get_data()

		#check overlap
		for i in np.arange(17):

			df.loc[r, 'Lesymap seed network #'] = p
			df.loc[r, 'Overlap with Yeo thalamus parcel #'] = i+1
			df.loc[r, 'zscore'] = np.mean(zmap[yeo_mask==i+1])
			r=r+1


	sns.catplot(x='Lesymap seed network #', y='zscore', hue='Overlap with Yeo thalamus parcel #', data=df, height=6, kind="bar", palette="muted")		


	### check overlap with yeo thalamus 7 network parcellation
	fn = '/home/kahwang/bsh/Corticothalamic_Network/Yeo_Networks_Analyses/Yeo_Networks/ThalamusParcellation/' + '1000subjects_TightThalamus_clusters007_ref_1mm.nii.gz'
	yeo_mask = nib.load(fn).get_data()

	#check mask
	for i in np.arange(7):
		print(np.sum(yeo_mask==i+1))

	#Check simulated score lesymap results overlap with Yeo's parcellation
	parcels = ['01' ,'02', '03', '04', '05', '06', '07']

	df = pd.DataFrame(columns=['Lesymap seed network #', 'FC zscore with Yeo thalamus parcel #', 'zscore'])
	r = 0 
	for p in parcels:
		fn = '/home/kahwang/bsh/Corticothalamic_Network/Yeo_Networks_Analyses/Network_Mapping/LESYMAP_zmaps/1mm/zmap_LESYMAP_Yeo_7_Net_%s_thr_1mm.nii.gz' %p
		zmap = nib.load(fn).get_data()

		#check overlap
		for i in np.arange(7):

			df.loc[r, 'Lesymap seed network #'] = p
			df.loc[r, 'FC zscore with Yeo thalamus parcel #'] = i+1
			df.loc[r, 'zscore'] = np.mean(zmap[yeo_mask==i+1])
			r=r+1


	sns.catplot(x='Lesymap seed network #', y='zscore', hue='FC zscore with Yeo thalamus parcel #', data=df, height=6, kind="bar", palette="muted")	



