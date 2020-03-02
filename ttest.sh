#!/bin/bash


for mask in CA001; do	#0902 1105 1692 1809 1830 2092 2105 2552 2781 3049 3184 CA018 CA041 CA085 CA104 CA105 CA134

	3dttest++ \
	-setA "/data/backed_up/shared/Tha_Lesion_Mapping/0*/seed_corr_${mask}_000_INDIV/WB_Z_ROI_001.nii.gz" \
	-prefix /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_${mask}.nii.gz


	
done