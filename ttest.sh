#!/bin/bash



### for lesion network mapping
# for mask in CA001; do	#0902 1105 1692 1809 1830 2092 2105 2552 2781 3049 3184 CA018 CA041 CA085 CA104 CA105 CA134 CA001

# 	3dttest++ \
# 	-setA "/data/backed_up/shared/Tha_Lesion_Mapping/0*/seed_corr_${mask}_000_INDIV/WB_Z_ROI_001.nii.gz" \
# 	-prefix /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_${mask}.nii.gz



# done



### for contrasting lesion network maps

# 3dttest++ \
# -setA MedTha \
# p1 /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_1830.nii.gz[0] \
# p2 /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_2105.nii.gz[0]  \
# p3 /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_3049.nii.gz[0] \
# p4 /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_2781.nii.gz[0]  \
# p5 /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_2552.nii.gz[0] \
# p6 /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_2092.nii.gz[0]  \
# p7 /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_CA104.nii.gz[0]  \
# p8 /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_CA105.nii.gz[0]  \
# p0 /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_1105.nii.gz[0]  \
# -setB LatTHa \
# p9 /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_1692.nii.gz[0]  \
# p10 /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_1809.nii.gz[0]  \
# p11 /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_3184.nii.gz[0]  \
# p12 /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_CA018.nii.gz[0]  \
# p13 /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_CA041.nii.gz[0]  \
# -prefix /data/backed_up/kahwang/Tha_Neuropsych/FC_analysis/ttest_TMTB_impaired \
# -resid /data/backed_up/kahwang/Tha_Neuropsych/FC_analysis/ttest_TMTB_impaired_resid.nii \
# -ACF



#3dClustSim -mask ttest_TMTB_impaired_resid.nii -acf 0.0890296 2.78616 12.4156


#-Clustsim 16 \
#-prefix_clustsim /data/backed_up/kahwang/Tha_Neuropsych/FC_analysis/ttest_TMTB_MedvLatTHa_clustsim



### Cortical lesymap seed FC map
#NKI
# group stats on lesion network
for mask in BNT_GM_Clust1 BNT_GM_Clust2 BNT_GM_Clust3 BNT_GM_Clust4 COM_FIG_RECALL_Clust1 COM_FIG_RECALL_Clust2 COM_FIG_RECALL_Clust3 COM_FIG_RECALL_Clust4 COWA_Clust1 COWA_Clust2 TMTB_Clust1 TMTB_Clust2; do

  rm /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_${mask}.nii.gz

	3dttest++ \
	-setA "/data/backed_up/shared/Tha_Lesion_Mapping/0*/seed_corr_${mask}_000_INDIV/WB_Z_ROI_00*.nii.gz" \
	-prefix /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_${mask}.nii.gz

done

# MGH
### group stats on lesion network
for mask in BNT_GM_Clust1 BNT_GM_Clust2 BNT_GM_Clust3 BNT_GM_Clust4 COM_FIG_RECALL_Clust1 COM_FIG_RECALL_Clust2 COM_FIG_RECALL_Clust3 COM_FIG_RECALL_Clust4 COWA_Clust1 COWA_Clust2 TMTB_Clust1 TMTB_Clust2; do

  rm /data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_${mask}.nii.gz

	3dttest++ \
	-setA "/data/backed_up/shared/Tha_Lesion_Mapping/MGH*/seed_corr_${mask}_000_INDIV/WB_Z_ROI_00*.nii.gz" \
	-prefix /data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_${mask}.nii.gz

done
