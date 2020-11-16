#!/bin/bash

##### Lesion network mapping
#Subjects=$(cat /data/backed_up/shared/Tha_Lesion_Mapping/Subject_List.txt)
#NKI
#$(cat /data/backed_up/shared/Tha_Lesion_Mapping/Subject_List.txt)

#0902 1105 1692 1809 1830 2092 2105 2552 2781 3049 3184 CA018 CA041 CA085 CA104 CA105 CA134
for subject in $(cat /data/backed_up/shared/Tha_Lesion_Mapping/Subject_List.txt); do

  rm -rf /data/backed_up/shared/Tha_Lesion_Mapping/${subject}/
	mkdir /data/backed_up/shared/Tha_Lesion_Mapping/${subject}/

	for mask in 1692 1809 1830 2105 3049 1105 2781 2552 2092 0902 3184 ca018 ca041 ca104 ca105 ca085 ca093 4036 4032 4041 4045; do

    rm /data/backed_up/shared/Tha_Lesion_Mapping/${subject}/rsmask_${mask}.nii.gz

		3dresample -master /data/backed_up/shared/NKI/${subject}/MNINonLinear/rfMRI_REST_mx_1400_ncsreg.nii.gz \
		-inset /home/kahwang/0.5mm/${mask}.nii.gz \
		-prefix /data/backed_up/shared/Tha_Lesion_Mapping/${subject}/rsmask_${mask}.nii.gz

		3dNetCorr -inset /data/backed_up/shared/NKI/${subject}/MNINonLinear/rfMRI_REST_mx_1400_ncsreg.nii.gz \
		-in_rois /data/backed_up/shared/Tha_Lesion_Mapping/${subject}/rsmask_${mask}.nii.gz \
		-nifti \
		-ts_wb_Z \
		-prefix /data/backed_up/shared/Tha_Lesion_Mapping/${subject}/seed_corr_${mask}


	done
done


#### group stats on lesion network
for mask in 1692 1809 1830 2105 3049 1105 2781 2552 2092 0902 3184 ca018 ca041 ca104 ca105 ca085 ca093 4036 4032 4041 4045; do

  rm /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_${mask}.nii.gz
	3dttest++ \
	-setA "/data/backed_up/shared/Tha_Lesion_Mapping/0*/seed_corr_${mask}_000_INDIV/WB_Z_ROI_001.nii.gz" \
	-prefix /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_${mask}.nii.gz



done




#### Use Lesymaps as seed

# TMTB_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/Trail_making_part_B_LESYMAP/stat_img.nii.gz')
# TMTB_LESYMAP_map = TMTB_LESYMAP.get_data()
# BNT_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/BOS_NAM_RAW/stat_img.nii.gz')
# BNT_LESYMAP_map = BNT_LESYMAP.get_data()
# COWA_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/MAE_COWA/stat_img.nii.gz')
# COWA_LESYMAP_map = COWA_LESYMAP.get_data()
# COM_FIG_COPY_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/CONS_CFT_RAW/stat_img.nii.gz')
# COM_FIG_COPY_LESYMAP_map = COM_FIG_COPY_LESYMAP.get_data()
# COM_FIG_RECALL_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/COM_FIG_RECALL/stat_img.nii.gz')
# COM_FIG_RECALL_LESYMAP_map = COM_FIG_RECALL_LESYMAP.get_data()
