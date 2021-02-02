from ThaHub import *
from nilearn.input_data import NiftiLabelsMasker
################################
## Calculate thalamocortical FC and thalamic vox by vox PC
################################

def generate_correlation_mat(x, y):
    """Correlate each n with each m.

    Parameters
    ----------
    x : np.array
      Shape N X T.

    y : np.array
      Shape M X T.

    Returns
    -------
    np.array
      N X M array in which each element is a correlation coefficient.

    """
    mu_x = x.mean(1)
    mu_y = y.mean(1)
    n = x.shape[1]
    if n != y.shape[1]:
        raise ValueError('x and y must ' +
                         'have the same number of timepoints.')
    s_x = x.std(1, ddof=n - 1)
    s_y = y.std(1, ddof=n - 1)
    cov = np.dot(x,
                 y.T) - n * np.dot(mu_x[:, np.newaxis],
                                  mu_y[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def pcorr_subcortico_cortical_connectivity(subcortical_ts, cortical_ts):
	''' function to do partial correlation bewteen subcortical and cortical ROI timeseries.
	Cortical signals (not subcortical) will be removed from subcortical and cortical ROIs,
	and then pairwise correlation will be calculated bewteen subcortical and cortical ROIs
	(but not between subcortico-subcortical or cortico-cortical ROIs).
	This partial correlation/regression approach is for cleaning subcortico-cortical
	conectivity, which seems to be heavily influenced by a global noise.
	usage: pcorr_mat = pcorr_subcortico-cortical(subcortical_ts, cortical_ts)
	----
	Parameters
	----
	subcortical_ts: txt file of timeseries data from subcortical ROIs/voxels, each row is an ROI
	cortical_ts: txt file of timeseries data from cortical ROIs, each row is an ROI
	pcorr_mat: output partial correlation matrix
	'''
	from scipy import stats, linalg
	from sklearn.decomposition import PCA

	# # transpose so that column is ROI, this is because output from 3dNetcorr is row-based.
	# subcortical_ts = subcortical_ts.T
	# cortical_ts = cortical_ts.T
	cortical_ts[np.isnan(cortical_ts)]=0
	subcortical_ts[np.isnan(subcortical_ts)]=0

	# check length of data
	assert cortical_ts.shape[0] == subcortical_ts.shape[0]
	num_vol = cortical_ts.shape[0]

	#first check that the dimension is appropriate
	num_cort = cortical_ts.shape[1]
	num_subcor = subcortical_ts.shape[1]
	num_total = num_cort + num_subcor

	#maximum number of regressors that we can use
	max_num_components = int(num_vol/20)
	if max_num_components > num_cort:
		max_num_components = num_cort-1

	pcorr_mat = np.zeros((num_total, num_total), dtype=np.float)

	for j in range(num_cort):
		k = np.ones(num_cort, dtype=np.bool)
		k[j] = False

		#use PCA to reduce cortical data dimensionality
		pca = PCA(n_components=max_num_components)
		pca.fit(cortical_ts[:,k])
		reduced_cortical_ts = pca.fit_transform(cortical_ts[:,k])

		#print("Amount of varaince explanined after PCA: %s" %np.sum(pca.explained_variance_ratio_))

		# fit cortical signal to cortical ROI TS, get betas
		beta_cortical = linalg.lstsq(reduced_cortical_ts, cortical_ts[:,j])[0]

		#get residuals
		res_cortical = cortical_ts[:, j] - reduced_cortical_ts.dot(beta_cortical)

		for i in range(num_subcor):
			# fit cortical signal to subcortical ROI TS, get betas
			beta_subcortical = linalg.lstsq(reduced_cortical_ts, subcortical_ts[:,i])[0]

			#get residuals
			res_subcortical = subcortical_ts[:, i] - reduced_cortical_ts.dot(beta_subcortical)

			#partial correlation
			pcorr_mat[i+num_cort, j] = stats.pearsonr(res_cortical, res_subcortical)[0]
			pcorr_mat[j,i+num_cort ] = pcorr_mat[i+num_cort, j]

	return pcorr_mat


### global variables, masks, files, etc.
# load files
MGH_fn = '/home/kahwang/bsh/MGH/MGH/*/MNINonLinear/rfMRI_REST_ncsreg.nii.gz'
MGH_files = glob.glob(MGH_fn)

NKI_fn = '/data/backed_up/shared/NKI/*/MNINonLinear/rfMRI_REST_mx_1400_ncsreg.nii.gz'
NKI_files = glob.glob(NKI_fn)
datafiles = [NKI_files, MGH_files]
datasets=['NKI', 'MGH']

# load masks
thalamus_mask = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz')
thalamus_mask_data = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_fdata()
thalamus_mask_data = thalamus_mask_data>0
thalamus_mask = nilearn.image.new_img_like(thalamus_mask, thalamus_mask_data)
mm_unique = nib.load('images/mm_unique.nii.gz')
mm_unique_2mm = resample_to_img(mm_unique, thalamus_mask, interpolation = 'nearest')
Schaefer400_mask = nib.load('/home/kahwang/bsh/ROIs/Schaefer400_7network_2mm.nii.gz')
cortex_masker = NiftiLabelsMasker(labels_img='/home/kahwang/bsh/ROIs/Schaefer400_7network_2mm.nii.gz', standardize=False)
Schaeffer_CI = np.loadtxt('/home/kahwang/bin/LesionNetwork/Schaeffer400_7network_CI')


def cal_PC():
    '''Calculate PC values '''
    for i, files in enumerate(datafiles):

        thresholds = [90,91,92,93,94,95,96,97,98]

        # saving both patial corr and full corr
        fpc_vectors = np.zeros((np.count_nonzero(thalamus_mask_data>0),len(files), len(thresholds)))
        ppc_vectors = np.zeros((np.count_nonzero(thalamus_mask_data>0),len(files), len(thresholds)))
        pc_vectors = [ppc_vectors, fpc_vectors]

        for ix, f in enumerate(files):
            functional_data = nib.load(f)
            #extract cortical ts from schaeffer 400 ROIs
            cortex_ts = cortex_masker.fit_transform(functional_data)
            #time by ROI
            #cortex_ts = cortex_ts.T
            #extract thalamus vox by vox ts
            thalamus_ts = masking.apply_mask(functional_data, thalamus_mask)
            # time by vox
            #thalamus_ts = thalamus_ts.T

            # concate, cortex + thalamus voxel, dimesnion should be 2627 (400 cortical ROIs plus 2227 thalamus voxel from morel atlas)
            # work on partial corr.
            #ts = np.concatenate((cortex_ts, thalamus_ts), axis=1)
            #corrmat = np.corrcoef(ts.T)
            pmat = pcorr_subcortico_cortical_connectivity(thalamus_ts, cortex_ts)
            thalamocortical_pfc = pmat[400:, 0:400]
            #extrat the thalamus by cortex FC matrix

            # fc marices
            thalamocortical_ffc = generate_correlation_mat(thalamus_ts.T, cortex_ts.T)
            #fcmats.append(thalamocortical_fc)
            #calculate PC with the extracted thalamocortical FC matrix

            #loop through threshold
            FCmats = [thalamocortical_pfc, thalamocortical_ffc]

            for j, thalamocortical_fc in enumerate(FCmats):
                for it, t in enumerate(thresholds):
                    temp_mat = thalamocortical_fc.copy()
                    temp_mat[temp_mat<np.percentile(temp_mat, t)] = 0
                    fc_sum = np.sum(temp_mat, axis=1)
                    kis = np.zeros(np.shape(fc_sum))

                    for ci in np.unique(Schaeffer_CI):
                        kis = kis + np.square(np.sum(temp_mat[:,np.where(Schaeffer_CI==ci)[0]], axis=1) / fc_sum)

                    pc_vectors[j][:,ix, it] = 1-kis

        fn = "data/%s_pc_vectors_pcorr" %datasets[i]
        np.save(fn, pc_vectors[0])
        fn = "data/%s_pc_vectors_corr" %datasets[i]
        np.save(fn, pc_vectors[1])

def cal_mmmask_FC():
    ''' calculate FC between corticla ROIs and thalamic mask of multitask impairment()'''

    for i, files in enumerate(datafiles):
        # save both pcorr and full corr
        fcpmat = np.zeros((np.count_nonzero(mm_unique_2mm.get_fdata()>0),400, len(files)))
        fcmat = np.zeros((np.count_nonzero(mm_unique_2mm.get_fdata()>0),400, len(files)))

        for ix, f in enumerate(files):
            functional_data = nib.load(f)
            #extract cortical ts from schaeffer 400 ROIs
            cortex_ts = cortex_masker.fit_transform(functional_data)
            #time by ROI
            #cortex_ts = cortex_ts.T
            #extract thalamus vox by vox ts
            thalamus_ts = masking.apply_mask(functional_data, mm_unique_2mm)
            # time by vox
            #thalamus_ts = thalamus_ts.T
            pmat = pcorr_subcortico_cortical_connectivity(thalamus_ts, cortex_ts)
            fcpmat[:,:,ix] = pmat[400:, 0:400]
            fcmat[:,:,ix] = generate_correlation_mat(thalamus_ts.T, cortex_ts.T)


        fn = "data/%s_mmmask_fc_pcorr" %datasets[i]
        np.save(fn, fcpmat)
        fn = "data/%s_mmmask_fc_fcorr" %datasets[i]
        np.save(fn, fcmat)



if __name__ == "__main__":

    cal_PC()
    #cal_mmmask_FC()








################################
# #### OLD backup code blocks
################################
# def run_LESYMAP_fc():
#     df = pd.read_csv('~/RDSS/tmp/data_z.csv')
#
# 	lesymap_clusters = ['BNT_GM_Clust1', 'BNT_GM_Clust2', 'BNT_GM_Clust3', 'BNT_GM_Clust4', 'COM_FIG_RECALL_Clust1', 'COM_FIG_RECALL_Clust2', 'COM_FIG_RECALL_Clust3', 'COM_FIG_RECALL_Clust4', 'COWA_Clust1', 'COWA_Clust2', 'TMTB_Clust1', 'TMTB_Clust2']
# 	fcdf = pd.DataFrame()
# 	i=0
# 	for p in df.loc[df['Site'] == 'Th']['Sub']:
# 		try:
# 			fn = '/home/kahwang/0.5mm/%s_2mm.nii.gz' %p
# 			m = nib.load(fn).get_data()
# 		except:
# 			continue
#
# 		for lesymap in lesymap_clusters: #MGH* or Sub* for NKI subjects. NKI not done yet.
# 			fn = '/data/backed_up/shared/Tha_Lesion_Mapping/MGH*/seed_corr_%s_ncsreg_000_INDIV/*.nii.gz' %lesymap
# 			files = glob.glob(fn)
#
# 			s = 0
# 			for f in files:
# 				fcmap = nib.load(f).get_data()
# 				fcdf.loc[i, 'Subject'] = str(s)
# 				fcdf.loc[i, 'Patient'] = p
# 				fcdf.loc[i, 'Cluster'] = lesymap
# 				fcdf.loc[i, 'FC'] = np.max(fcmap[m>0])
# 				fcdf.loc[i, 'TMTB_z_Impaired'] = df.loc[df['Sub'] == p]['TMTB_z_Impaired'].values[0]
# 				fcdf.loc[i, 'BNT_z_Impaired'] = df.loc[df['Sub'] == p]['BNT_z_Impaired'].values[0]
# 				fcdf.loc[i, 'COWA_z_Impaired'] = df.loc[df['Sub'] == p]['COWA_z_Impaired'].values[0]
# 				fcdf.loc[i, 'Complex_Figure_Recall_z_Impaired'] = df.loc[df['Sub'] == p]['Complex_Figure_Recall_z_Impaired'].values[0]
# 				fcdf.loc[i, 'TMTB_z'] = df.loc[df['Sub'] == p]['TMTB_z'].values[0] * -1 # invert
# 				fcdf.loc[i, 'BNT_z'] = df.loc[df['Sub'] == p]['BNT_z'].values[0]
# 				fcdf.loc[i, 'COWA_z'] = df.loc[df['Sub'] == p]['COWA_z'].values[0]
# 				fcdf.loc[i, 'Complex_Figure_Recall_z'] = df.loc[df['Sub'] == p]['Complex_Figure_Recall_z'].values[0]
#
# 				if lesymap == 'BNT_GM_Clust1':
# 					fcdf.loc[i, 'Task'] = 'BNT'
# 					fcdf.loc[i, 'zscore'] = df.loc[df['Sub'] == p]['BNT_z'].values[0]
# 				if lesymap == 'BNT_GM_Clust2':
# 					fcdf.loc[i, 'Task'] = 'BNT'
# 					fcdf.loc[i, 'zscore'] = df.loc[df['Sub'] == p]['BNT_z'].values[0]
# 				if lesymap == 'BNT_GM_Clust3':
# 					fcdf.loc[i, 'Task'] = 'BNT'
# 					fcdf.loc[i, 'zscore'] = df.loc[df['Sub'] == p]['BNT_z'].values[0]
# 				if lesymap == 'BNT_GM_Clust4':
# 					fcdf.loc[i, 'Task'] = 'BNT'
# 					fcdf.loc[i, 'zscore'] = df.loc[df['Sub'] == p]['BNT_z'].values[0]
# 				if lesymap == 'COM_FIG_RECALL_Clust1':
# 					fcdf.loc[i, 'Task'] = 'COM_FIG_RECALL'
# 					fcdf.loc[i, 'zscore'] = df.loc[df['Sub'] == p]['Complex_Figure_Recall_z'].values[0]
# 				if lesymap == 'COM_FIG_RECALL_Clust2':
# 					fcdf.loc[i, 'Task'] = 'COM_FIG_RECALL'
# 					fcdf.loc[i, 'zscore'] = df.loc[df['Sub'] == p]['Complex_Figure_Recall_z'].values[0]
# 				if lesymap == 'COM_FIG_RECALL_Clust3':
# 					fcdf.loc[i, 'Task'] = 'COM_FIG_RECALL'
# 					fcdf.loc[i, 'zscore'] = df.loc[df['Sub'] == p]['Complex_Figure_Recall_z'].values[0]
# 				if lesymap == 'COM_FIG_RECALL_Clust4':
# 					fcdf.loc[i, 'Task'] = 'COM_FIG_RECALL'
# 					fcdf.loc[i, 'zscore'] = df.loc[df['Sub'] == p]['Complex_Figure_Recall_z'].values[0]
# 				if lesymap == 'COWA_Clust1':
# 					fcdf.loc[i, 'Task'] = 'COWA'
# 					fcdf.loc[i, 'zscore'] = df.loc[df['Sub'] == p]['COWA_z'].values[0]
# 				if lesymap == 'COWA_Clust2':
# 					fcdf.loc[i, 'Task'] = 'COWA'
# 					fcdf.loc[i, 'zscore'] = df.loc[df['Sub'] == p]['COWA_z'].values[0]
# 				if lesymap == 'TMTB_Clust1':
# 					fcdf.loc[i, 'Task'] = 'TMTB'
# 					fcdf.loc[i, 'zscore'] = df.loc[df['Sub'] == p]['TMTB_z'].values[0] * -1
# 				if lesymap == 'TMTB_Clust2':
# 					fcdf.loc[i, 'Task'] = 'TMTB'
# 					fcdf.loc[i, 'zscore'] = df.loc[df['Sub'] == p]['TMTB_z'].values[0] * -1
# 				i = i+1
# 				s = s+1
#
# 	fcdf.to_csv('~/RDSS/tmp/fcdatav2.csv')
#
# def pcorr_subcortico_cortical_connectivity(subcortical_ts, cortical_ts):
# 	''' function to do partial correlation bewteen subcortical and cortical ROI timeseries.
# 	Cortical signals (not subcortical) will be removed from subcortical and cortical ROIs,
# 	and then pairwise correlation will be calculated bewteen subcortical and cortical ROIs
# 	(but not between subcortico-subcortical or cortico-cortical ROIs).
# 	This partial correlation/regression approach is for cleaning subcortico-cortical
# 	conectivity, which seems to be heavily influenced by a global noise.
# 	usage: pcorr_mat = pcorr_subcortico-cortical(subcortical_ts, cortical_ts)
# 	----
# 	Parameters
# 	----
# 	subcortical_ts: txt file of timeseries data from subcortical ROIs/voxels, each row is an ROI
# 	cortical_ts: txt file of timeseries data from cortical ROIs, each row is an ROI
# 	pcorr_mat: output partial correlation matrix
# 	'''
# 	from scipy import stats, linalg
# 	from sklearn.decomposition import PCA
#
# 	# # transpose so that column is ROI, this is because output from 3dNetcorr is row-based.
# 	# subcortical_ts = subcortical_ts.T
# 	# cortical_ts = cortical_ts.T
# 	cortical_ts[np.isnan(cortical_ts)]=0
# 	subcortical_ts[np.isnan(subcortical_ts)]=0
#
# 	# check length of data
# 	assert cortical_ts.shape[0] == subcortical_ts.shape[0]
# 	num_vol = cortical_ts.shape[0]
#
# 	#first check that the dimension is appropriate
# 	num_cort = cortical_ts.shape[1]
# 	num_subcor = subcortical_ts.shape[1]
# 	num_total = num_cort + num_subcor
#
# 	#maximum number of regressors that we can use
# 	max_num_components = int(num_vol/20)
# 	if max_num_components > num_cort:
# 		max_num_components = num_cort-1
#
# 	pcorr_mat = np.zeros((num_total, num_total), dtype=np.float)
#
# 	for j in range(num_cort):
# 		k = np.ones(num_cort, dtype=np.bool)
# 		k[j] = False
#
# 		#use PCA to reduce cortical data dimensionality
# 		pca = PCA(n_components=max_num_components)
# 		pca.fit(cortical_ts[:,k])
# 		reduced_cortical_ts = pca.fit_transform(cortical_ts[:,k])
#
# 		#print("Amount of varaince explanined after PCA: %s" %np.sum(pca.explained_variance_ratio_))
#
# 		# fit cortical signal to cortical ROI TS, get betas
# 		beta_cortical = linalg.lstsq(reduced_cortical_ts, cortical_ts[:,j])[0]
#
# 		#get residuals
# 		res_cortical = cortical_ts[:, j] - reduced_cortical_ts.dot(beta_cortical)
#
# 		for i in range(num_subcor):
# 			# fit cortical signal to subcortical ROI TS, get betas
# 			beta_subcortical = linalg.lstsq(reduced_cortical_ts, subcortical_ts[:,i])[0]
#
# 			#get residuals
# 			res_subcortical = subcortical_ts[:, i] - reduced_cortical_ts.dot(beta_subcortical)
#
# 			#partial correlation
# 			pcorr_mat[i+num_cort, j] = stats.pearsonr(res_cortical, res_subcortical)[0]
# 			pcorr_mat[j,i+num_cort ] = pcorr_mat[i+num_cort, j]
#
# 	return pcorr_mat
#
#
# if __name__ == "__main__":
#
# 	########################################################################
# 	# Fig 3, LESYMAP lesion network
# 	########################################################################
#
# 	########################################################################
#
# 	# Plot the LESYMAP results
# 	TMTB_LESYMAP = nib.load('/home/kahwang/Trail_B_and_A_Difference/stat_img.nii.gz')
# 	fn = '/home/kahwang/RDSS/tmp/TMTB_LESYMAP_nii.png'
# 	plotting.plot_stat_map(TMTB_LESYMAP, colorbar=False, title = 'Trail Making B', output_file = fn)
#
# 	BNT_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/BOS_NAM_RAW/stat_img.nii.gz')
# 	fn = '/home/kahwang/RDSS/tmp/BNT_LESYMAP_nii.png'
# 	plotting.plot_stat_map(BNT_LESYMAP, colorbar=False, title = 'Boston Naming', output_file = fn)
#
# 	COM_FIG_RECALL_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/COM_FIG_RECALL/stat_img.nii.gz')
# 	fn = '/home/kahwang/RDSS/tmp/COM_FIG_RECALL_nii.png'
# 	plotting.plot_stat_map(COM_FIG_RECALL_LESYMAP, colorbar=False, title = 'Complex Figure Recall', output_file = fn)
#
# 	#COM_FIG_RECOG_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/COM_FIG_RECOG/stat_img.nii.gz')
# 	#fn = '/home/kahwang/RDSS/tmp/COM_FIG_RECOG_nii.png'
# 	#plotting.plot_stat_map(COM_FIG_RECOG_LESYMAP, colorbar=False, title = 'Complex Figure Recognition', output_file = fn)
#
# 	#COM_FIG_RECOG_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/COM_FIG_RECOG/stat_img.nii.gz')
# 	#fn = '/home/kahwang/RDSS/tmp/COM_FIG_RECOG_nii.png'
# 	#plotting.plot_stat_map(COM_FIG_RECOG_LESYMAP, colorbar=False, title = 'Complex Figure Recognition', output_file = fn)
#
# 	COWA_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/MAE_COWA/stat_img.nii.gz')
# 	fn = '/home/kahwang/RDSS/tmp/COWA_nii.png'
# 	plotting.plot_stat_map(COWA_LESYMAP, colorbar=False, title = 'COWA', output_file = fn)
#
# 	REY5_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/REY_5/stat_img.nii.gz')
# 	fn = '/home/kahwang/RDSS/tmp/REY5_nii.png'
# 	plotting.plot_stat_map(REY5_LESYMAP, colorbar=False, title = 'RAVL Trial 5', output_file = fn)
#
#
#
# 	########################################################################
# 	# Plot the LESYMAP grey matter mask clusters
# 	# First load all the lesymap, and get rid of white matter overlap
#
# 	# use MNI 2mm template as base
# 	m = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz')
#
# 	#load WM atlas
# 	WM_atlas = nib.load('/home/kahwang/bsh/standard/mni_icbm152_nlin_asym_09c/wm_2mm.nii').get_data() > 0.5
#
# 	#load each LESYMAP. Ones not loaded don't have GM clusters. Including RAVL tests
# 	TMTB_LESYMAP_map = nib.load('/home/kahwang/Trail_B_and_A_Difference/stat_img_2mm.nii.gz').get_data()!=0
# 	BNT_LESYMAP_map = nib.load('/home/kahwang/LESYMAP_for_Kai/BOS_NAM_RAW/stat_img_2mm.nii.gz').get_data()!=0
# 	COWA_LESYMAP_map = nib.load('/home/kahwang/LESYMAP_for_Kai/MAE_COWA/stat_img_2mm.nii.gz').get_data()!=0
# 	COM_FIG_RECALL_LESYMAP_map = nib.load('/home/kahwang/LESYMAP_for_Kai/COM_FIG_RECALL/stat_img_2mm.nii.gz').get_data()!=0
#
# 	# get rid of WM
# 	TMTB_LESYMAP_map = 1* ((1 * TMTB_LESYMAP_map - 1 * WM_atlas) > 0)
# 	BNT_LESYMAP_map = 1* ((1 * BNT_LESYMAP_map - 1 * WM_atlas) > 0)
# 	COWA_LESYMAP_map = 1* ((1 * COWA_LESYMAP_map - 1 * WM_atlas) > 0)
# 	COM_FIG_RECALL_LESYMAP_map = 1* ((1 * COM_FIG_RECALL_LESYMAP_map - 1 * WM_atlas) > 0)
#
# 	TMTB_LESYMAP_GM_nii = nilearn.image.new_img_like(m, TMTB_LESYMAP_map, copy_header=True)
# 	BNT_LESYMAP_GM_nii = nilearn.image.new_img_like(m, BNT_LESYMAP_map, copy_header=True)
# 	COWA_LESYMAP_GM_nii = nilearn.image.new_img_like(m, COWA_LESYMAP_map, copy_header=True)
# 	COM_FIG_RECALL_LESYMAP_GM_nii = nilearn.image.new_img_like(m, COM_FIG_RECALL_LESYMAP_map, copy_header=True)
# 	TMTB_LESYMAP_GM_nii.to_filename('/home/kahwang/LESYMAP_for_Kai/TMTB_GM.nii.gz')
#
# 	mni_template = nib.load('/data/backed_up/shared/standard/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_brain.nii')
#
# 	img = plotting.plot_stat_map(TMTB_LESYMAP_GM_nii, bg_img = mni_template, display_mode='z', cut_coords=4, colorbar = False, black_bg=False, cmap='bwr')
# 	img.savefig('/home/kahwang/RDSS/tmp/TMTB_LESYMAP_GM_nii.png')
# 	#plotting.show()
# 	img = plotting.plot_stat_map(BNT_LESYMAP_GM_nii, bg_img = mni_template, display_mode='z', cut_coords=4, colorbar = False, black_bg=False, cmap='bwr')
# 	img.savefig('/home/kahwang/RDSS/tmp/BNT_LESYMAP_GM_nii.png')
# 	# plotting.show()
# 	img =plotting.plot_stat_map(COWA_LESYMAP_GM_nii, bg_img = mni_template, display_mode='z', cut_coords=4, colorbar = False, black_bg=False, cmap='bwr')
# 	img.savefig('/home/kahwang/RDSS/tmp/COWA_LESYMAP_GM_nii.png')
# 	#plotting.show()
# 	img =plotting.plot_stat_map(COM_FIG_RECALL_LESYMAP_GM_nii, bg_img = mni_template, display_mode='z', cut_coords=4, colorbar = False, black_bg=False, cmap='bwr')
# 	img.savefig('/home/kahwang/RDSS/tmp/COM_FIG_RECALL_LESYMAP_GM_nii.png')
# 	#plotting.show()
#
# 	# Plot overlap. Looks like COWA and BNT has overlapping clusters in LIFG, duh.
# 	LESSYMAP_GM_overlap = TMTB_LESYMAP_map + BNT_LESYMAP_map + COWA_LESYMAP_map + COM_FIG_RECALL_LESYMAP_map
# 	LESSYMAP_GM_overlap_nii = nilearn.image.new_img_like(m, LESSYMAP_GM_overlap, copy_header = True)
# 	plotting.plot_stat_map(LESSYMAP_GM_overlap_nii, bg_img = mni_template, display_mode='z', cut_coords=8, colorbar = True, black_bg=False, cmap='bwr', output_file = '/home/kahwang/RDSS/tmp/Overlap_LESYMAP_GM_nii.png')
# 	#plotting.show()
#
#
# 	#############################################
# 	### Need to run Prep_LESYMAP_clusters.sh first
# 	### Use linear mixed effect regression model to test FC differences
#
# 	# this will generate the fcdf
# 	#run_LESYMAP_fc()
#
#
# 	########### plot lessyamp_FC in thalamus
# 	MGH_groupFC_BNT_GM_Clust1_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_BNT_GM_Clust1_ncsreg.nii.gz').get_data()
# 	MGH_groupFC_BNT_GM_Clust2_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_BNT_GM_Clust2_ncsreg.nii.gz').get_data()
# 	MGH_groupFC_BNT_GM_Clust3_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_BNT_GM_Clust3_ncsreg.nii.gz').get_data()
# 	MGH_groupFC_BNT_GM_Clust4_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_BNT_GM_Clust4_ncsreg.nii.gz').get_data()
# 	MGH_groupFC_COM_FIG_RECALL_Clust1_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_COM_FIG_RECALL_Clust1_ncsreg.nii.gz').get_data()
# 	MGH_groupFC_COM_FIG_RECALL_Clust2_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_COM_FIG_RECALL_Clust2_ncsreg.nii.gz').get_data()
# 	MGH_groupFC_COM_FIG_RECALL_Clust3_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_COM_FIG_RECALL_Clust3_ncsreg.nii.gz').get_data()
# 	MGH_groupFC_COM_FIG_RECALL_Clust4_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_COM_FIG_RECALL_Clust4_ncsreg.nii.gz').get_data()
# 	MGH_groupFC_COWA_Clust1_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_COWA_Clust1_ncsreg.nii.gz').get_data()
# 	MGH_groupFC_COWA_Clust2_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_COWA_Clust2_ncsreg.nii.gz').get_data()
# 	MGH_groupFC_TMTB_Clust1_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_TMTB_Clust1_ncsreg.nii.gz').get_data()
# 	MGH_groupFC_TMTB_Clust2_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_TMTB_Clust2_ncsreg.nii.gz').get_data()
#
# 	# create cluster FC average for each task
# 	groupFC_BNT = (MGH_groupFC_BNT_GM_Clust1_ncsreg + MGH_groupFC_BNT_GM_Clust2_ncsreg + MGH_groupFC_BNT_GM_Clust3_ncsreg + MGH_groupFC_BNT_GM_Clust4_ncsreg) / 4
# 	groupFC_COM_FIG_RECALL = (MGH_groupFC_COM_FIG_RECALL_Clust1_ncsreg + MGH_groupFC_COM_FIG_RECALL_Clust2_ncsreg + MGH_groupFC_COM_FIG_RECALL_Clust3_ncsreg + MGH_groupFC_COM_FIG_RECALL_Clust4_ncsreg) / 4
# 	groupFC_COWA = (MGH_groupFC_COWA_Clust1_ncsreg + MGH_groupFC_COWA_Clust2_ncsreg ) / 2
# 	groupFC_TMTB = (MGH_groupFC_TMTB_Clust1_ncsreg + MGH_groupFC_TMTB_Clust2_ncsreg ) / 2
#
# 	h = MGH_groupFC_BNT_GM_Clust1_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_BNT_GM_Clust1_ncsreg.nii.gz')
# 	groupFC_BNT_nii = nilearn.image.new_img_like(h, groupFC_BNT)
# 	groupFC_COM_FIG_RECALL_nii = nilearn.image.new_img_like(h, groupFC_COM_FIG_RECALL)
# 	groupFC_COWA_nii = nilearn.image.new_img_like(h, groupFC_COWA)
# 	groupFC_TMTB_nii = nilearn.image.new_img_like(h, groupFC_TMTB)
#
# 	#mask thalamus FC
# 	thalamus_mask = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz')
# 	thalamus_mask_data = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_data()
# 	thalamus_mask_data = thalamus_mask_data>0
# 	thalamus_mask = nilearn.image.new_img_like(thalamus_mask, thalamus_mask_data)
#
# 	names = ['BNT_FC.nii.gz', 'COM_FIG_RECALL_FC.nii.gz', 'COWA_FC.nii.gz', 'TMTB_FC.nii.gz']
# 	new_nii=[]
# 	for i, nii in enumerate([groupFC_BNT_nii, groupFC_COM_FIG_RECALL_nii, groupFC_COWA_nii,  groupFC_TMTB_nii]):
# 		vox_vals = masking.apply_mask(nii, thalamus_mask)[1][0] # t brik
# 		vox_rvals = masking.apply_mask(nii, thalamus_mask)[0][0] # r value
# 		vox_rvals[vox_vals < 4.13] = 0 #np.percentile(vox_vals,90) #threshold at t =4.13, which is p=1-e5
# 		new_nii.append(masking.unmask(vox_rvals, thalamus_mask))
# 		new_nii[i].to_filename(names[i])
# 		#plotting.plot_stat_map(new_nii[i], bg_img = mni_template, display_mode='z', cut_coords=4, colorbar = False, black_bg=False, cmap='bwr')
# 		#plotting.show()
#
# 		#find overlaps
# 		if i == 0:
# 			overlapping_voxels = np.zeros(vox_vals.shape)
# 			vox_rvals[vox_rvals>0] = 1
# 		else:
# 			vox_rvals[vox_rvals>0] = 1
# 			overlapping_voxels = overlapping_voxels + vox_rvals
#
# 	LESSY_FC_Overlap = masking.unmask(overlapping_voxels, thalamus_mask)
# 	LESSY_FC_Overlap.to_filename('LESSY_FC_Overlap.nii.gz')
# 	#plotting.plot_stat_map(LESSY_FC_Overlap, bg_img = mni_template, display_mode='z', cut_coords=4, colorbar = True, black_bg=False, cmap='bwr' output_file = '')
# 	#plotting.show()
#
#
# 	########### fit lme models
# 	# remember to run run_LESYMAP_fc()
# 	df = pd.read_csv('~/RDSS/tmp/data_z.csv')
# 	#run_LESYMAP_fc(df)
# 	#fcdf = pd.read_csv('~/RDSS/tmp/fcdata.csv')
#
# 	# sumarize data frame across normative subjects
# 	fcdf = pd.read_csv('~/RDSS/tmp/fcdatav2.csv')
# 	gdf = fcdf.groupby(['Patient', 'Cluster', 'Task']).mean().reset_index()
# 	#gdf = gdf.loc[gdf['Task']!='COM_FIG_RECALL']
#
# 	# full model
# 	vc = {'Patient': '0 + C(Patient)',}
# 	model = smf.mixedlm("zscore ~ FC *Task", gdf.dropna(), groups=gdf.dropna()['Patient']).fit()
# 	print(model.summary())
# 	#sns.lmplot( data=gdf, x="FC", y="zscore", hue="Task")
# 	#plt.show()
#
# 	#not mlm
# 	#smf.ols("zscore ~ FC *Task", gdf.dropna()).fit().summary()
#
# 	# treat each subject as random effect
# 	vc = {'Subject': '0 + C(Subject)'}
# 	#fcdf['groups'] = 1
# 	model = smf.mixedlm("zscore ~ FC*Task", fcdf.dropna() , vc_formula = vc, re_formula = '~1', groups=fcdf.dropna()['Subject']).fit()
# 	model.summary()
#
# 	# statsmodel, subject with random intercept, random slope for each subj
# 	# https://www.statsmodels.org/stable/mixed_linear.html,
# 	# linear relation, whether zscore has a relationship with FC
# 	# the nature of the relation within  was assessed with a linear mixed effects linear regression
# 	# and showed that (peak: β = −0.013, 95% CI = [−0.019, −0.006], P < 0.001).
#
# 	tdf = fcdf.loc[fcdf['Task'] == 'TMTB']
# 	md_TMTb = smf.mixedlm("TMTB_z ~ FC", tdf.dropna() , re_formula="FC",groups=tdf.dropna()['Subject']).fit()
# 	print(md_TMTb.summary())
#
# 	tdf = fcdf.loc[fcdf['Task'] == 'COWA']
# 	# md = smf.mixedlm("FC ~  + COWA_z_Impaired", tdf, groups=tdf['Subject']).fit()
# 	# print(md.summary())
# 	md_COWA = smf.mixedlm("COWA_z ~ FC", tdf.dropna(), re_formula="FC",groups=tdf.dropna()['Subject']).fit()
# 	print(md_COWA.summary())
#
# 	tdf = fcdf.loc[fcdf['Task'] == 'COM_FIG_RECALL']
# 	# md = smf.mixedlm("FC ~  + Complex_Figure_Recall_z_Impaired", tdf, groups=tdf['Subject']).fit()
# 	# print(md.summary())
# 	md_CF = smf.mixedlm("Complex_Figure_Recall_z ~  FC", tdf.dropna(), re_formula="FC",groups=tdf.dropna()['Subject']).fit()
# 	print(md_CF.summary())
#
# 	tdf = fcdf.loc[fcdf['Task'] == 'BNT']
# 	# md = smf.mixedlm("FC ~  + BNT_z_Impaired", tdf, groups=tdf['Subject']).fit()
# 	# print(md.summary())
# 	md_BNT = smf.mixedlm("BNT_z ~  FC", tdf.dropna(), re_formula="FC",groups=tdf.dropna()['Subject']).fit()
# 	print(md_BNT.summary())
#
# 	#### plot coef and CI
# 	coef_df = pd.DataFrame()
# 	coef_df.loc[0, 'Coefficients'] = md_TMTb.params['FC']
# 	coef_df.loc[0, 'errors'] = md_TMTb.params['FC'] - md_TMTb.conf_int()[0]['FC']
# 	coef_df.loc[1, 'Coefficients'] = md_COWA.params['FC']
# 	coef_df.loc[1, 'errors'] = md_COWA.params['FC'] - md_COWA.conf_int()[0]['FC']
# 	coef_df.loc[2, 'Coefficients'] = md_BNT.params['FC']
# 	coef_df.loc[2, 'errors'] = md_BNT.params['FC'] - md_BNT.conf_int()[0]['FC']
# 	coef_df.loc[3, 'Coefficients'] = md_CF.params['FC']
# 	coef_df.loc[3, 'errors'] = md_CF.params['FC'] - md_CF.conf_int()[0]['FC']
# 	coef_df.loc[0, 'Task']	= 'TMT part B'
# 	coef_df.loc[1, 'Task']	= 'COWA'
# 	coef_df.loc[2, 'Task']	= 'Boston Naming'
# 	coef_df.loc[3, 'Task']	= 'Complex Figure Recall'
#
# 	coef_df.plot(x='Task', y='Coefficients', xerr= 'errors', kind='barh', legend = False, figsize=(4,2))
# 	plt.xlabel('Coefficient Estimate')
# 	#plt.tight_layout()
# 	fn = '/home/kahwang/RDSS/tmp/fcregcoef.pdf'
# 	#plt.savefig(fn)
#
#
#
# 	########################################################################
# 	# Calculate participation coef for each lesion mask,
# 	# Determine nuclei overlap for each lesion mask.
# 	########################################################################
#
# 	#### PC from the JN paper, calcuated based on resting-state networks
# 	def cal_morel_lesion_overlap():
# 		PC_map = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/FC_analysis/PC.nii.gz')
#
# 		#nuclei
# 		Morel = {
# 		1: 'AN',
# 		2: 'VM',
# 		3: 'VL',
# 		4: 'MGN',
# 		5: 'MD',
# 		6: 'PuA',
# 		7: 'LP',
# 		8: 'IL',
# 		9: 'VA',
# 		10: 'Po',
# 		11: 'LGN',
# 		12: 'PuM',
# 		13: 'PuI',
# 		14: 'PuL',
# 		17: 'VP'
# 		}
#
# 		morel_atlas = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_data()
#
# 		for p in df.loc[df['Site'] == 'Th']['Sub'] :
#
# 			if p == '4045':
# 				continue #no mask yet
#
# 			else:
# 				#resample mask from .5 mm grid to 2 mm grid
# 				#cmd = "3dresample -master /data/backed_up/kahwang/Tha_Neuropsych/FC_analysis/PC.nii.gz -inset /home/kahwang/0.5mm/%s.nii.gz -prefix /home/kahwang/0.5mm/%s_2mm.nii.gz" %(p, p)
# 				#os.system(cmd)
#
# 				fn = '/home/kahwang/0.5mm/%s_2mm.nii.gz' %p
# 				lesion_mask = nib.load(fn)
#
# 				df.loc[df['Sub']==p,'PC'] = np.mean(PC_map.get_data()[lesion_mask.get_data()>0])
#
# 				for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]:
# 					df.loc[df['Sub']==p, Morel[n]] = 8 * np.sum(lesion_mask.get_data()[morel_atlas==n])
#
# 		print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['MM_impaired']>=2)]['PC'].values, df.loc[(df['Site']=='Th') & (df['MM_impaired']<2)]['PC'].values))
# 		print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th')]['MM_impaired'].values, df.loc[(df['Site']=='ctx')]['MM_impaired'].values))
#
# 		df.to_csv('~/RDSS/tmp/data_z.csv')
#
# 	#### calculate PC subject by subject, so we can fit a linear mixed effect model
#
	# def cal_indiv_rsPC():
	# 	# calculate PC subject by subject
	# 	thalamus_mask = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz')
	# 	thalamus_mask_data = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_data()
	# 	thalamus_mask_data = thalamus_mask_data>0
	# 	thalamus_mask = nilearn.image.new_img_like(thalamus_mask, thalamus_mask_data, copy_header = True)
	# 	Schaefer400_mask = nib.load('/home/kahwang/bsh/ROIs/Schaefer400_7network_2mm.nii.gz')
    #
	# 	from nilearn.input_data import NiftiLabelsMasker
		# cortex_masker = NiftiLabelsMasker(labels_img='/home/kahwang/bsh/ROIs/Schaefer400_7network_2mm.nii.gz', standardize=False)
        #
		# Schaeffer_CI = np.loadtxt('/home/kahwang/bin/LesionNetwork/Schaeffer400_7network_CI')
        #
		# fn = '/home/kahwang/bsh/MGH/MGH/*/MNINonLinear/rfMRI_REST_ncsreg.nii.gz'
		# files = glob.glob(fn)
        #
		# pc_vectors = np.zeros((np.count_nonzero(thalamus_mask_data>0),len(files)))
		# #fcmats = []
        #
		# for ix, f in enumerate(files):
		# 	functional_data = nib.load(f)
        #
		# 	#extract cortical ts from schaeffer 400 ROIs
		# 	cortex_ts = cortex_masker.fit_transform(functional_data)
        #
		# 	#extract thalamus vox by vox ts
		# 	thalamus_ts = masking.apply_mask(functional_data, thalamus_mask)
        #
		# 	# concate, cortex + thalamus voxel, dimesnion should be 2627 (400 cortical ROIs plus 2227 thalamus voxel from morel atlas)
		# 	# work on partial corr.
		# 	#ts = np.concatenate((cortex_ts, thalamus_ts), axis=1)
		# 	#corrmat = np.corrcoef(ts.T)
		# 	pmat = pcorr_subcortico_cortical_connectivity(thalamus_ts, cortex_ts)
		# 	#extrat the thalamus by cortex FC matrix
		# 	thalamocortical_fc = pmat[400:, 0:400]
		# 	#fcmats.append(thalamocortical_fc)
		# 	#calculate PC with the extracted thalamocortical FC matrix
		# 	thalamocortical_fc[thalamocortical_fc<np.percentile(thalamocortical_fc, 90)] = 0
		# 	fc_sum = np.sum(thalamocortical_fc, axis=1)
		# 	kis = np.zeros(np.shape(fc_sum))
		# 	for ci in np.unique(Schaeffer_CI):
		# 		kis = kis + np.square(np.sum(thalamocortical_fc[:,np.where(Schaeffer_CI==ci)[0]], axis=1) / fc_sum)
        #
		# 	pc_vectors[:,ix] = 1-kis
        #
		# np.save('pc_vectors_pcorr', pc_vectors)

#
# 	########################################################################
# 	# Calculate lesymap FC's participation coef, using different lesymap seed FC as "communities"
# 	########################################################################
#
# 	def cal_indiv_LESYMAP_PC():
# 		thalamus_mask = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz')
# 		thalamus_mask_data = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_data()
# 		thalamus_mask_data = thalamus_mask_data>0
# 		thalamus_mask = nilearn.image.new_img_like(thalamus_mask, thalamus_mask_data, copy_header = True)
# 		files = glob.glob('/data/backed_up/shared/Tha_Lesion_Mapping/MGH*/seed_corr_BNT_GM_Clust1_ncsreg_000_INDIV/*.nii.gz')
#
# 		# FCMaps
# 		lesymap_clusters = ['BNT_GM_Clust1', 'BNT_GM_Clust2', 'BNT_GM_Clust3', 'BNT_GM_Clust4', 'COM_FIG_RECALL_Clust1', 'COM_FIG_RECALL_Clust2', 'COM_FIG_RECALL_Clust3', 'COM_FIG_RECALL_Clust4', 'COWA_Clust1', 'COWA_Clust2', 'TMTB_Clust1', 'TMTB_Clust2']
# 		# How FC Maps are grouped into task indices
# 		lesymap_CI = np.array([1, 1, 1 , 1, 2, 2, 2, 2, 3, 3, 4, 4])
#
# 		# extra thalamic voxel by lesymap seed FC vectors, save.
# 		fc_vectors = np.zeros((np.count_nonzero(thalamus_mask_data>0),len(lesymap_CI), len(files)))
# 		for il, lesymap in enumerate(lesymap_clusters):
# 			fn = '/data/backed_up/shared/Tha_Lesion_Mapping/MGH*/seed_corr_%s_ncsreg_000_INDIV/*.nii.gz' %lesymap
# 			files = glob.glob(fn)
# 			for ix, f in enumerate(files):
# 				fcmap = nib.load(f)
# 				fc_vectors[:,il, ix] = masking.apply_mask(fcmap, thalamus_mask)
#
# 		np.save('fc_vectors', fc_vectors)
#
#
# 		#### Calculate sub by sub PC with lesmap seed FC vector, then do mixed effect regression
# 		# load lesymap seed FC vectors
# 		fc_vectors = np.load('fc_vectors.npy')
# 		fc_vectors[fc_vectors<0] = 0
#
# 		#lessymap FC's PC calculation
# 		fc_sum = np.sum(fc_vectors, axis = 1)
# 		kis = np.zeros(np.shape(fc_sum))
# 		for ci in np.array([1, 2, 3, 4]):
# 			kis = kis + np.square(np.sum(fc_vectors[:,np.where(lesymap_CI==ci)[0],:], axis=1) / fc_sum)
# 		fc_pc = 1-kis
# 		np.save('fc_pc', fc_pc)
#
#
#
# 	def cal_groupave_LESYMAP_FC_PC():
# 		''' Calculate subs average lesymap FC PC'''
#
# 		thalamus_mask = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz')
# 		thalamus_mask_data = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_data()
# 		thalamus_mask_data = thalamus_mask_data>0
# 		thalamus_mask = nilearn.image.new_img_like(thalamus_mask, thalamus_mask_data, copy_header = True)
#
# 		MGH_groupFC_BNT_GM_Clust1_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_BNT_GM_Clust1_ncsreg.nii.gz')
# 		MGH_groupFC_BNT_GM_Clust2_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_BNT_GM_Clust2_ncsreg.nii.gz')
# 		MGH_groupFC_BNT_GM_Clust3_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_BNT_GM_Clust3_ncsreg.nii.gz')
# 		MGH_groupFC_BNT_GM_Clust4_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_BNT_GM_Clust4_ncsreg.nii.gz')
# 		MGH_groupFC_COM_FIG_RECALL_Clust1_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_COM_FIG_RECALL_Clust1_ncsreg.nii.gz')
# 		MGH_groupFC_COM_FIG_RECALL_Clust2_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_COM_FIG_RECALL_Clust2_ncsreg.nii.gz')
# 		MGH_groupFC_COM_FIG_RECALL_Clust3_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_COM_FIG_RECALL_Clust3_ncsreg.nii.gz')
# 		MGH_groupFC_COM_FIG_RECALL_Clust4_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_COM_FIG_RECALL_Clust4_ncsreg.nii.gz')
# 		MGH_groupFC_COWA_Clust1_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_COWA_Clust1_ncsreg.nii.gz')
# 		MGH_groupFC_COWA_Clust2_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_COWA_Clust2_ncsreg.nii.gz')
# 		MGH_groupFC_TMTB_Clust1_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_TMTB_Clust1_ncsreg.nii.gz')
# 		MGH_groupFC_TMTB_Clust2_ncsreg = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_TMTB_Clust2_ncsreg.nii.gz')
#
# 		fcmaps = [MGH_groupFC_BNT_GM_Clust1_ncsreg, MGH_groupFC_BNT_GM_Clust2_ncsreg, MGH_groupFC_BNT_GM_Clust3_ncsreg, MGH_groupFC_BNT_GM_Clust4_ncsreg,
# 		MGH_groupFC_COM_FIG_RECALL_Clust1_ncsreg, MGH_groupFC_COM_FIG_RECALL_Clust2_ncsreg, MGH_groupFC_COM_FIG_RECALL_Clust3_ncsreg, MGH_groupFC_COM_FIG_RECALL_Clust4_ncsreg,
# 		MGH_groupFC_COWA_Clust1_ncsreg, MGH_groupFC_COWA_Clust2_ncsreg, MGH_groupFC_TMTB_Clust1_ncsreg, MGH_groupFC_TMTB_Clust2_ncsreg]
#
# 		lesymap_CI = np.array([1, 1, 1 , 1, 2, 2, 2, 2, 3, 3, 4, 4])
#
# 		fc_vectors = np.zeros((np.count_nonzero(thalamus_mask_data>0),len(lesymap_CI)))
# 		t_vectors = np.zeros((np.count_nonzero(thalamus_mask_data>0),len(lesymap_CI)))
# 		for im, fcmap in enumerate(fcmaps):
# 			fc_vectors[:,im] = masking.apply_mask(fcmap, thalamus_mask)[0][0]
# 			t_vectors[:,im] = masking.apply_mask(fcmap, thalamus_mask)[1][0]
#
# 		fc_vectors[t_vectors<2.13] = 0
# 		fc_sum = np.sum(fc_vectors, axis = 1)
# 		kis = np.zeros(np.shape(fc_sum))
# 		for ci in np.array([1, 2, 3, 4]):
# 			kis = kis + np.square(np.sum(fc_vectors[:,np.where(lesymap_CI==ci)[0]], axis=1) / fc_sum)
# 		meanfc_pc = 1-kis
# 		np.save('meanLESYfc_pc', meanfc_pc)
# 		meanlcpc_image = masking.unmask(meanfc_pc-.5, thalamus_mask) # clip at .5
# 		meanlcpc_image.to_filename('LESSY_FC_PC.nii.gz')
#
#
#
# 	def write_group_PC_df():
# 		''' compilie dataframe for group average PC and lesymap PC'''
# 		meanfc_pc = np.load('meanLESYfc_pc.npy')
# 		thalamus_mask = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz')
# 		thalamus_mask_data = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_data()
# 		thalamus_mask_data = thalamus_mask_data>0
# 		thalamus_mask = nilearn.image.new_img_like(thalamus_mask, thalamus_mask_data, copy_header = True)
# 		meanlcpc_image = masking.unmask(meanfc_pc, thalamus_mask).get_data()
# 		rsfc_pc = nib.load('RSFC_PC.nii.gz').get_fdata()
#
# 		lspcdf = pd.DataFrame()
# 		i=0
# 		for p in df.loc[df['Site'] == 'Th']['Sub']:
# 			try:
# 				fn = '/home/kahwang/0.5mm/%s_2mm.nii.gz' %p
# 				m = nib.load(fn).get_data()
# 			except:
# 				continue
# 			lspcdf.loc[i, 'Patient'] = p
# 			lspcdf.loc[i, 'LESYMAP_PC'] = np.nanmean(meanlcpc_image[np.where(m>0)][meanlcpc_image[np.where(m>0)]>0])
# 			lspcdf.loc[i, 'RSFC_PC'] = np.nanmean(rsfc_pc[np.where(m>0)][rsfc_pc[np.where(m>0)]>0])
# 			lspcdf.loc[i, 'TMTB_z'] = df.loc[df['Sub'] == p]['TMTB_z'].values[0] * -1
# 			lspcdf.loc[i, 'BNT_z'] = df.loc[df['Sub'] == p]['BNT_z'].values[0]
# 			lspcdf.loc[i, 'COWA_z'] = df.loc[df['Sub'] == p]['COWA_z'].values[0]
# 			lspcdf.loc[i, 'Complex_Figure_Recall_z'] = df.loc[df['Sub'] == p]['Complex_Figure_Recall_z'].values[0]
# 			lspcdf.loc[i, 'LesionSize'] = np.sum(m>0) * 8
# 			lspcdf.loc[i, 'MM_impaired'] =  df.loc[df['Sub'] == p]['MM_impaired'].values[0]
# 			i=i+1
# 		#lspcdf['MM_Impaired_num'] = 1*(lspcdf['COWA_z']<-1) + 1*(lspcdf['TMTB_z']<-1) + 1*(lspcdf['BNT_z']<-1) + 1*(lspcdf['Complex_Figure_Recall_z']<-1)
#
#
#
#
# 		#meanlcpc_image_nii = nilearn.image.new_img_like(thalamus_mask, meanlcpc_image)
# 		#meanlcpc_image_nii.to_filename('LESSY_FC_PC.nii.gz')
#
# 	pcdf = pd.read_csv('~/RDSS/tmp/pcdf.csv')
# 	md = smf.mixedlm("LESYMAP_PC ~ MM_impaired", pcdf, groups=pcdf['Subject']).fit()
# 	print(md.summary())
#
# 	#gdf = pcdf.groupby(['Patient']).mean()
# 	lspcdf['load'] = lspcdf['RSFC_PC'] * lspcdf['LesionSize']
#
# 	smf.ols("MM_impaired ~ load", lspcdf).fit().summary()
# 	#scipy.stats.mannwhitneyu(gdf.loc[(gdf['MM_impaired']==1)]['meanLESYMAP_PC'].values, gdf.loc[(gdf['MM_impaired']==0)]['meanLESYMAP_PC'].values)
#
# 	sns.lmplot( data=lspcdf, x="load", y="MM_impaired")
# 	plt.show()
#
# 	md = smf.mixedlm("MM_impaired_num ~ LESYMAP_PC", pcdf, groups=pcdf['Subject']).fit()
# 	print(md.summary())
#
# 	md = smf.mixedlm("MM_impaired ~ PC", pcdf, groups=pcdf['Subject']).fit()
# 	print(md.summary())
#
# 	md = smf.mixedlm("MM_impaired_num ~ PC", pcdf, groups=pcdf['Subject']).fit()
# 	print(md.summary())
#
# 	#pc_image = masking.unmask(np.nanmean(fc_pc, axis=1), thalamus_mask)
# 	#plotting.plot_stat_map(pc_image, display_mode='z', cut_coords=12, colorbar = True, black_bg=False, cmap='ocean_hot')
# 	#plotting.show()
#
#
# 	#### Check if lesymap FC's participation coef correlate with resting-state network's PC
# 	#print(np.corrcoef(np.nanmean(pc_vectors, axis=1),np.nanmean(fc_pc, axis=1)))
#
#
# 	####################################################
# 	# Compare the voxel-wise PC values between patients with and witout MM impairment
# 	PC_map = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/FC_analysis/PC.nii.gz')
# 	thalamus_mask = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz')
# 	thalamus_mask_data = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_data()
# 	thalamus_mask_data = thalamus_mask_data>0
# 	thalamus_mask = nilearn.image.new_img_like(thalamus_mask, thalamus_mask_data)
#
# 	vox_PC_df = pd.DataFrame()
#
# 	i=0
# 	for p in df.loc[df['Site'] == 'Th']['Sub']:
# 		try:
# 			fn = '/home/kahwang/0.5mm/%s_2mm.nii.gz' %p
# 			m = nib.load(fn)
# 		except:
# 			continue
#
# 		pcs = masking.apply_mask(PC_map, m)
# 		pcs = pcs[pcs>0]
#
# 		for ix in np.arange(0, pcs.size):
# 			vox_PC_df.loc[i, 'PC'] = pcs[ix]
# 			vox_PC_df.loc[i, 'Patient'] = p
# 			vox_PC_df.loc[i, 'MM_impaired_num'] = df.loc[df['Sub'] == p]['MM_impaired'].values[0]
# 			vox_PC_df.loc[i, 'MM_impaired'] = int(df.loc[df['Sub'] == p]['MM_impaired'].values[0]>2)
# 			i = i+1
#
# 	# md = smf.ols("PC ~ MM_impaired_num", vox_PC_df).fit()
# 	# print(md.summary())
#
# ########################################################################
# # White matter tracktography
# ########################################################################
#
# 	def cal_WM_density():
# 		WM_densities = ['BNT', 'COM_FIG_RECALL', 'CONS_CFT', 'COWA', 'TMTB']
#
# 		for i, p in enumerate(df.loc[df['Site'] == 'Th']['Sub']):
#
# 			#WM density is in 1mm gtid, so resample lesion mask
# 			#cmd = "3dresample -master /home/kahwang/LESYMAP_for_Kai/BNT_WMTrack.nii.gz -inset /home/kahwang/0.5mm/%s.nii.gz -prefix /home/kahwang/0.5mm/%s_1mm.nii.gz" %(p, p)
# 			#os.system(cmd)
#
# 			try:
# 				fn = '/home/kahwang/0.5mm/%s_1mm.nii.gz' %p
# 				m = nib.load(fn).get_data()
#
# 			except:
# 				continue
#
# 			for wm in WM_densities:
# 				fn = '/home/kahwang/LESYMAP_for_Kai/%s_WMTrack.nii.gz' %wm
# 				wm_density = nib.load(fn).get_data()
# 				if any(wm_density[np.where(m>0)]):
# 					df.loc[i, wm + '_WMtrack'] = 1.0
# 				else:
# 					df.loc[i, wm + '_WMtrack'] = 0.0
#
# 				df[wm + '_WMtrack'] = df[wm + '_WMtrack'].astype('float')
#
# 		df.to_csv('~/RDSS/tmp/data_z.csv')
#
# 	# scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['BNT_WMtrack']>0)]['BNT_z'].values, df.loc[(df['Site']=='Th') & (df['BNT_WMtrack']==0)]['BNT_z'].values)
# 	# scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['COM_FIG_RECALL_WMtrack']>0)]['Complex_Figure_Recall_z'].values, df.loc[(df['Site']=='Th') & (df['COM_FIG_RECALL_WMtrack']==0)]['Complex_Figure_Recall_z'].values)
# 	# scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['CONS_CFT_WMtrack']>0)]['Complex_Figure_Copy_z'].values, df.loc[(df['Site']=='Th') & (df['CONS_CFT_WMtrack']==0)]['Complex_Figure_Copy_z'].values)
# 	# scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['COWA_WMtrack']>0)]['COWA_z'].values, df.loc[(df['Site']=='Th') & (df['COWA_WMtrack']==0)]['COWA_z'].values)
# 	# scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['TMTB_WMtrack']>0)]['TMTB_z'].values, df.loc[(df['Site']=='Th') & (df['TMTB_WMtrack']==0)]['TMTB_z'].values)
#
#
#
#
# 	################################
# 	# compare voxel distribution of lesymap FC weights for different tasks, for each lesion mask
# 	################################
# 	#use these nii objects to calcuate FC weights for each task
# 	#groupFC_BNT_nii
# 	#groupFC_COM_FIG_RECALL_nii
# 	#groupFC_COWA_nii
# 	#groupFC_TMTB_nii
#
# 	#Patients lesion disruption
# 	### patients with MM impairments (>3):
# 	# 2105 2552 2092 ca085 ca093 ca104 ca105 1692 1830 3049
# 	# 1692: COWA, RAVLT recall, learn
# 	# 1830: TMTB RAVLT Recog, learn
# 	# 3049: TMTB COWA, learn
# 	# 2105: TMTB, COWA, RVLT recall,
# 	# 2552: TMTB, BNT, COWA, RVLT recall
# 	# 2092: TMTB, COWA, RVLT recall, RVLT recog, RVLT, learn.
# 	# ca085: BNT, RVLT recall, Com Figure Copy, Fig COM_FIG_RECALL
# 	# CA093, BNT, RVLT recog, RVLT learn, com fig copy, com fig COM_FIG_RECAL
# 	# ca104, TMTB, RVLT recall, RVLT recog, RVLT learn, com fig copy, com fig recall
# 	# ca105, TMTB, COWA, RVLT recall, RVLT recog, RVLT learn, com fig copy, com fig recal
#
# 	def plot_voxel_FC_dist():
# 		# seems like the best way is to first calculation the ratio of FC / total FC weight, and plot the voxel wise ditribution of this ratio. You would expect most be around .5 (for 2 tasks) or .3 (for 3 tasks)
# 		#lesymap_clusters = ['BNT_GM_Clust1', 'BNT_GM_Clust2', 'BNT_GM_Clust3', 'BNT_GM_Clust4', 'COM_FIG_RECALL_Clust1', 'COM_FIG_RECALL_Clust2', 'COM_FIG_RECALL_Clust3', 'COM_FIG_RECALL_Clust4', 'COWA_Clust1', 'COWA_Clust2', 'TMTB_Clust1', 'TMTB_Clust2']
#
# 		lesymap_clusters_p={}
# 		lesymap_clusters_p['2105'] = [groupFC_COWA_nii, groupFC_TMTB_nii, groupFC_BNT_nii, groupFC_COM_FIG_RECALL_nii]
# 		lesymap_clusters_p['2552'] = [groupFC_COWA_nii, groupFC_TMTB_nii, groupFC_BNT_nii, groupFC_COM_FIG_RECALL_nii]
# 		lesymap_clusters_p['2092'] = [groupFC_COWA_nii, groupFC_TMTB_nii, groupFC_BNT_nii, groupFC_COM_FIG_RECALL_nii]
# 		lesymap_clusters_p['ca085'] = [groupFC_COWA_nii, groupFC_TMTB_nii, groupFC_BNT_nii, groupFC_COM_FIG_RECALL_nii]
# 		lesymap_clusters_p['ca093'] = [groupFC_COWA_nii, groupFC_TMTB_nii, groupFC_BNT_nii, groupFC_COM_FIG_RECALL_nii]
# 		lesymap_clusters_p['ca104'] = [groupFC_COWA_nii, groupFC_TMTB_nii, groupFC_BNT_nii, groupFC_COM_FIG_RECALL_nii]
# 		lesymap_clusters_p['ca105'] = [groupFC_COWA_nii, groupFC_TMTB_nii, groupFC_BNT_nii, groupFC_COM_FIG_RECALL_nii]
# 		lesymap_clusters_p['3049'] = [groupFC_COWA_nii, groupFC_TMTB_nii, groupFC_BNT_nii, groupFC_COM_FIG_RECALL_nii]
#
# 		#tasks={}
# 		tasks = ['COWA', 'TMTB', 'BNT', 'Recall']
#
# 		vwdf = pd.DataFrame()
#
# 		for p in ['ca105']:
# 			try:
# 				fn = '/home/kahwang/0.5mm/%s_2mm.nii.gz' %p
# 				m = nib.load(fn).get_data()
# 			except:
# 				continue
#
#
# 			fcsum=0
# 			for lesymap in lesymap_clusters_p[p]:
# 				#fcfile = '/home/kahwang/bsh/Tha_Lesion_Mapping/MGH_groupFC_%s_ncsreg.nii.gz'  %lesymap
# 				fcmap = lesymap.get_data()[:,:,:,0,0][m>0]
# 				fcmap[fcmap<0] = 0
# 				fcsum = fcsum+abs(fcmap)
#
# 			tempdf=	pd.DataFrame()
# 			for i, lesymap in enumerate(lesymap_clusters_p[p]):
# 				ttdf = pd.DataFrame()
# 				#fcfile = '/home/kahwang/bsh/Tha_Lesion_Mapping/MGH_groupFC_%s_ncsreg.nii.gz'  %lesymap
# 				fcmap = lesymap.get_data()[:,:,:,0,0][m>0]
# 				fcmap[fcmap<0] = 0
# 				ttdf['weight'] = abs(fcmap)/fcsum
# 				ttdf['task'] = tasks[i]
# 				ttdf['subject'] = p
#
# 				tempdf = pd.concat([ttdf,tempdf])
# 			vwdf = pd.concat([tempdf,vwdf])
#
# 		sdf = vwdf.loc[vwdf['subject']=='ca105']
# 		#sdf = sdf.loc[sdf['task'].isin(['COWA_Clust1', 'COWA_Clust2', 'TMTB_Clust1', 'TMTB_Clust2'])]
# 		sns.histplot(data=sdf, x="weight", hue='task')
# 		plt.show()
#
#
# 	################################
# 	# compare go group averaged PC values
# 	################################
#
# 	def cal_ave_pcorr():
# 		# calculate the average partial thalamocortical FC
#
# 		thalamus_mask = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz')
# 		thalamus_mask_data = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_data()
# 		thalamus_mask_data = thalamus_mask_data>0
# 		thalamus_mask = nilearn.image.new_img_like(thalamus_mask, thalamus_mask_data, copy_header = True)
# 		Schaefer400_mask = nib.load('/home/kahwang/bsh/ROIs/Schaefer400_7network_2mm.nii.gz')
# 		Schaeffer_CI = np.loadtxt('/home/kahwang/bin/LesionNetwork/Schaeffer400_7network_CI')
#
# 		from nilearn.input_data import NiftiLabelsMasker
# 		cortex_masker = NiftiLabelsMasker(labels_img='/home/kahwang/bsh/ROIs/Schaefer400_7network_2mm.nii.gz', standardize=False)
#
# 		fn = '/home/kahwang/bsh/MGH/MGH/*/MNINonLinear/rfMRI_REST_ncsreg.nii.gz'
# 		files = glob.glob(fn)
#
# 		#pc_vectors = np.zeros((np.count_nonzero(thalamus_mask_data>0),len(files)))
# 		fcmats = []
#
# 		for ix, f in enumerate(files):
# 			functional_data = nib.load(f)
#
# 			#extract cortical ts from schaeffer 400 ROIs
# 			cortex_ts = cortex_masker.fit_transform(functional_data)
#
# 			#extract thalamus vox by vox ts
# 			thalamus_ts = masking.apply_mask(functional_data, thalamus_mask)
#
# 			# concate, cortex + thalamus voxel, dimesnion should be 2627 (400 cortical ROIs plus 2227 thalamus voxel from morel atlas)
# 			# work on partial corr.
# 			#ts = np.concatenate((cortex_ts, thalamus_ts), axis=1)
# 			#corrmat = np.corrcoef(ts.T)
# 			pmat = pcorr_subcortico_cortical_connectivity(thalamus_ts, cortex_ts)
# 			#extrat the thalamus by cortex FC matrix
# 			thalamocortical_fc = pmat[400:, 0:400]
# 			fcmats.append(thalamocortical_fc)
# 			#calculate PC with the extracted thalamocortical FC matrix
#
# 		#PC calculation
# 		avemat = np.nanmean(fcmats, axis = 0)
# 		np.save('ave_pcorrmat', avemat)
# 		pcorr_pc = []
# 		for ix, thresh in enumerate(np.arange(85, 100, 1)):
# 			tempmap = avemat.copy()
# 			tempmap[tempmap<np.percentile(tempmap, thresh)] = 0
# 			fc_sum = np.sum(tempmap, axis=1)
# 			kis = np.zeros(np.shape(fc_sum))
# 			for ci in np.unique(Schaeffer_CI):
# 				kis = kis + np.square(np.sum(tempmap[:,np.where(Schaeffer_CI==ci)[0]], axis=1) / fc_sum)
#
# 			pcorr_pc.append(1-kis)
#
# 		pcorr_pc = np.array(pcorr_pc)
# 		np.save('pc_vectors_avepcorr', pcorr_pc)
#
# 	pcorr_pc = np.load('pc_vectors_avepcorr.npy')
# 	pc_image = masking.unmask(np.nanmean(pcorr_pc, axis=0), thalamus_mask)
# 	pc_image.to_filename('RSFC_PC.nii.gz')
# 	plotting.plot_stat_map(pc_image, display_mode='z', cut_coords=12, colorbar = True, black_bg=False, cmap='ocean_hot', vmax=0.4)
# 	plotting.show()
#


    #######
    #######
    ####### LEFT OVERS
    #######
    #######

    #
    # # a = nilearn.image.new_img_like(m, TMTB_LESYMAP_map, copy_header=True)
    # # a.to_filename('test.nii')
    # #plotting.plot_glass_brain(resample_from_to(TMTB_LESYMAP, m), threshold=0.8)
    # #plotting.show()
    #
    # for p in df.loc[df['Site'] == 'Th']['Sub']:
    #
    # 	if p == '4045':
    # 		continue #no mask yet
    # 	else:
    # 		fcfile = '/home/kahwang/bsh/Tha_Lesion_Mapping/MGH_groupFC_%s.nii.gz'  %p
    # 		fcmap = nib.load(fcfile).get_data()[:,:,:,0,1]
    #
    # 		df.loc[df['Sub'] == p, 'TMTB_FC'] = np.mean(fcmap[TMTB_LESYMAP_map >0])
    # 		df.loc[df['Sub'] == p, 'BNT_FC'] = np.mean(fcmap[BNT_LESYMAP_map >0])
    # 		df.loc[df['Sub'] == p, 'COWA_FC'] = np.mean(fcmap[COWA_LESYMAP_map >0])
    # 		df.loc[df['Sub'] == p, 'COM_FIG_COPY_FC'] = np.mean(fcmap[COM_FIG_COPY_LESYMAP_map >0])
    # 		df.loc[df['Sub'] == p, 'COM_FIG_RECALL_FC'] = np.mean(fcmap[COM_FIG_RECALL_LESYMAP_map >0])
    #
    # print(df.groupby(['TMTB_z_Impaired'])['TMTB_FC'].mean())
    # print(df.groupby(['BNT_z_Impaired'])['BNT_FC'].mean())
    # print(df.groupby(['COWA_z_Impaired'])['COWA_FC'].mean())
    # print(df.groupby(['Complex_Figure_Copy_z_Impaired'])['COM_FIG_COPY_FC'].mean())
    # print(df.groupby(['Complex_Figure_Recall_z_Impaired'])['COM_FIG_RECALL_FC'].mean())
    #
    # scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['TMTB_z_Impaired']==True)]['TMTB_FC'].values, df.loc[(df['Site']=='Th') & (df['TMTB_z_Impaired']==False)]['TMTB_FC'].values)
    # scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==True)]['BNT_FC'].values, df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==False)]['BNT_FC'].values)
    # scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['COWA_z_Impaired']==True)]['COWA_FC'].values, df.loc[(df['Site']=='Th') & (df['COWA_z_Impaired']==False)]['COWA_FC'].values)
    # scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['Complex_Figure_Copy_z_Impaired']==True)]['COM_FIG_COPY_FC'].values, df.loc[(df['Site']=='Th') & (df['Complex_Figure_Copy_z_Impaired']==False)]['COM_FIG_COPY_FC'].values)
    # scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==True)]['COM_FIG_RECALL_FC'].values, df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==False)]['COM_FIG_RECALL_FC'].values)
    #
    #
    #
    # ### run Prep_LESYMAP_clusters.sh, this one use cortical lesion masks, separated by GM, as seeds
    # lesymap_clusters = ['BNT_GM_Clust1', 'BNT_GM_Clust2', 'BNT_GM_Clust3', 'BNT_GM_Clust4', 'COM_FIG_RECALL_Clust1', 'COM_FIG_RECALL_Clust2', 'COM_FIG_RECALL_Clust3', 'COM_FIG_RECALL_Clust4', 'COWA_Clust1', 'COWA_Clust2', 'TMTB_Clust1', 'TMTB_Clust2']
    #
    # for lesymap in lesymap_clusters:
    # 	fcfile = '/home/kahwang/bsh/Tha_Lesion_Mapping/MGH_groupFC_%s_ncsreg.nii.gz'  %lesymap
    # 	fcmap = nib.load(fcfile).get_data()[:,:,:,0,1]
    #
    # 	for p in df.loc[df['Site'] == 'Th']['Sub']:
    # 		try:
    # 			fn = '/home/kahwang/0.5mm/%s_2mm.nii.gz' %p
    # 			m = nib.load(fn).get_data()
    # 		except:
    # 			continue
    #
    # 		df.loc[df['Sub'] == p, lesymap] = np.mean(fcmap[m>0])
    #
    # print(df.groupby(['TMTB_z_Impaired'])['TMTB_Clust1'].mean())
    # print(df.groupby(['TMTB_z_Impaired'])['TMTB_Clust2'].mean())
    # print(df.groupby(['BNT_z_Impaired'])['BNT_GM_Clust1'].mean())
    # print(df.groupby(['BNT_z_Impaired'])['BNT_GM_Clust2'].mean())
    # print(df.groupby(['BNT_z_Impaired'])['BNT_GM_Clust3'].mean())
    # print(df.groupby(['BNT_z_Impaired'])['BNT_GM_Clust4'].mean())
    # print(df.groupby(['COWA_z_Impaired'])['COWA_Clust1'].mean())
    # print(df.groupby(['COWA_z_Impaired'])['COWA_Clust2'].mean())
    # print(df.groupby(['Complex_Figure_Recall_z_Impaired'])['COM_FIG_RECALL_Clust1'].mean())
    # print(df.groupby(['Complex_Figure_Recall_z_Impaired'])['COM_FIG_RECALL_Clust2'].mean())
    # print(df.groupby(['Complex_Figure_Recall_z_Impaired'])['COM_FIG_RECALL_Clust3'].mean())
    # print(df.groupby(['Complex_Figure_Recall_z_Impaired'])['COM_FIG_RECALL_Clust4'].mean())
    #
    # scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['TMTB_z_Impaired']==True)]['TMTB_Clust1'].values, df.loc[(df['Site']=='Th') & (df['TMTB_z_Impaired']==False)]['TMTB_Clust1'].values)
    # scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['TMTB_z_Impaired']==True)]['TMTB_Clust2'].values, df.loc[(df['Site']=='Th') & (df['TMTB_z_Impaired']==False)]['TMTB_Clust2'].values)
    #
    # scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==True)]['BNT_GM_Clust1'].values, df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==False)]['BNT_GM_Clust1'].values)
    # scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==True)]['BNT_GM_Clust2'].values, df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==False)]['BNT_GM_Clust2'].values)
    # scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==True)]['BNT_GM_Clust3'].values, df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==False)]['BNT_GM_Clust3'].values)
    # scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==True)]['BNT_GM_Clust4'].values, df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==False)]['BNT_GM_Clust4'].values)
    #
    # scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==True)]['COM_FIG_RECALL_Clust1'].values, df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==False)]['COM_FIG_RECALL_Clust1'].values)
    # scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==True)]['COM_FIG_RECALL_Clust2'].values, df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==False)]['COM_FIG_RECALL_Clust2'].values)
    # scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==True)]['COM_FIG_RECALL_Clust3'].values, df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==False)]['COM_FIG_RECALL_Clust3'].values)
    # scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==True)]['COM_FIG_RECALL_Clust4'].values, df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==False)]['COM_FIG_RECALL_Clust4'].values)


    #resample LESYMAP output from 1mm grid to 2mm grid (FC maps in 2mm grid)
    # os.system("3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz -inset /home/kahwang/LESYMAP_for_Kai/Trail_making_part_B_LESYMAP/stat_img.nii.gz -prefix /home/kahwang/LESYMAP_for_Kai/Trail_making_part_B_LESYMAP/stat_img_2mm.nii.gz")
    # os.system("3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz -inset /home/kahwang/LESYMAP_for_Kai/BOS_NAM_RAW/stat_img.nii.gz -prefix /home/kahwang/LESYMAP_for_Kai/BOS_NAM_RAW/stat_img_2mm.nii.gz")
    # os.system("3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz -inset /home/kahwang/LESYMAP_for_Kai/MAE_COWA/stat_img.nii.gz -prefix /home/kahwang/LESYMAP_for_Kai/MAE_COWA/stat_img_2mm.nii.gz")
    # os.system("3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz -inset /home/kahwang/LESYMAP_for_Kai/CONS_CFT_RAW/stat_img.nii.gz -prefix /home/kahwang/LESYMAP_for_Kai/CONS_CFT_RAW/stat_img_2mm.nii.gz")
    # os.system("3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz -inset /home/kahwang/LESYMAP_for_Kai/COM_FIG_RECALL/stat_img.nii.gz -prefix /home/kahwang/LESYMAP_for_Kai/COM_FIG_RECALL/stat_img_2mm.nii.gz")
    # os.system("3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz -inset /home/kahwang/bsh/standard/mni_icbm152_nlin_asym_09c/mni_icbm152_wm_tal_nlin_asym_09c_2mm.nii -prefix /home/kahwang/bsh/standard/mni_icbm152_nlin_asym_09c/wm_2mm.nii")

	#### lmplot
	# melt
	# tdf = pd.melt(fcdf, id_vars = ['Subject', 'FC'],
	# 	value_vars = ['TMTB_z', 'BNT_z', 'COWA_z', 'Complex_Figure_Recall_z'], value_name = 'Z Score', var_name ='Task')
	#
	# g = sns.lmplot(x="FC", y="Z Score", col="Task", hue='Task', sharex = False,
	#                data=tdf, height=6, aspect=.4, x_jitter=.1, ci=None)
	# #g.set(ylim=(-0.1, 0.1))
	# plt.show()
