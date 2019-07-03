



# for sub in $(cat /home/kahwang/Trail_making_part_B_LESYMAP/subList_No_Epilepsy); do


# 	3dresample -inset /home/kahwang/Trail_making_part_B_LESYMAP/Masks/${sub}.nii.gz \
# 		-prefix /home/kahwang/Trail_making_part_B_LESYMAP/Masks/${sub}_2mm.nii.gz \
# 		-master /data/backed_up/shared/Tha_Lesion_Mapping/NKI_ttest_0902.nii.gz

# done

	

	
for mask in 0902 1105 1692 1809 1830 2092 2105 2552 2697 2781 3049 3184; do		
	
	rm /home/kahwang/Tha_Lesion_Masks/LesionLoad/LesionLoad_sum_mask${mask}
	touch /home/kahwang/Tha_Lesion_Masks/LesionLoad/LesionLoad_sum_mask${mask}
	
	for sub in $(cat /home/kahwang/Trail_making_part_B_LESYMAP/subList_No_Epilepsy); do
		#3dmaskave -quiet -mask /home/kahwang/Trail_making_part_B_LESYMAP/Masks/${sub}_2mm.nii.gz \
		#/data/backed_up/shared/Tha_Lesion_Mapping/NKI_ttest_${mask}.nii.gz[0] \
		#>> /home/kahwang/Tha_Lesion_Masks/LesionLoad/LesionLoad_mask${mask}

		3dROIstats -nzsum -quiet -nomeanout -mask /home/kahwang/Trail_making_part_B_LESYMAP/Masks/${sub}_2mm.nii.gz \
		/data/backed_up/shared/Tha_Lesion_Mapping/NKI_ttest_${mask}.nii.gz[0] >> /home/kahwang/Tha_Lesion_Masks/LesionLoad/LesionLoad_sum_mask${mask}
	done	


done