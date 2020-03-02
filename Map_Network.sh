#!/bin/bash


#Subjects=$(cat /data/backed_up/shared/Tha_Lesion_Mapping/Subject_List.txt)
#NKI
#$(cat /data/backed_up/shared/Tha_Lesion_Mapping/Subject_List.txt)

#0902 1105 1692 1809 1830 2092 2105 2552 2781 3049 3184 CA018 CA041 CA085 CA104 CA105 CA134
for subject in $(cat /data/backed_up/shared/Tha_Lesion_Mapping/Subject_List.txt); do
 
	mkdir /data/backed_up/shared/Tha_Lesion_Mapping/${subject}/
	
	for mask in CA001; do	

		3dresample -master /data/backed_up/shared/NKI/${subject}/MNINonLinear/rfMRI_REST_mx_1400_ncsreg.nii.gz \
		-inset /data/backed_up/kahwang/Tha_Neuropsych/Lesion_Masks/${mask}.nii.gz \
		-prefix /data/backed_up/shared/Tha_Lesion_Mapping/${subject}/rsmask_${mask}.nii.gz 

		3dNetCorr -inset /data/backed_up/shared/NKI/${subject}/MNINonLinear/rfMRI_REST_mx_1400_ncsreg.nii.gz \
		-in_rois /data/backed_up/shared/Tha_Lesion_Mapping/${subject}/rsmask_${mask}.nii.gz \
		-nifti \
		-ts_wb_Z \
		-prefix /data/backed_up/shared/Tha_Lesion_Mapping/${subject}/seed_corr_${mask}


		# 3dmaskave -q -mask /home/kahwang/Tha_Lesion_Masks/${mask}_2mm.nii.gz \
		# 	/data/backed_up/shared/NKI/${subject}/MNINonLinear/rfMRI_REST_mx_1400_ncsreg.nii.gz \
		# 	 > /data/backed_up/shared/Tha_Lesion_Mapping/${subject}/${mask}_${subject}_TS.1D
		
		# 3dfim+ -input /data/backed_up/shared/NKI/${subject}/MNINonLinear/rfMRI_REST_mx_1400_ncsreg.nii.gz \
		# 	-ideal_file /data/backed_up/shared/Tha_Lesion_Mapping/${subject}/${mask}_${subject}_TS.1D \
		# 	-out 'Correlation' \
		# 	-bucket /data/backed_up/shared/Tha_Lesion_Mapping/${subject}/${mask}_${subject}_TS_fit.nii.gz \


	done
done



