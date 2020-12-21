# sript to process the LESYMAP clusters
# separate WM and Gray Matter Voxels.
# Then us AFNI GUI to label clusters.


# resample to 2x2x2
# 3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz \
# -inset /home/kahwang/LESYMAP_for_Kai/BNT.nii.gz -prefix /home/kahwang/LESYMAP_for_Kai/BNT_2mm.nii.gz
#
# 3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz \
# -inset /home/kahwang/LESYMAP_for_Kai/COM_FIG_RECALL.nii.gz -prefix /home/kahwang/LESYMAP_for_Kai/COM_FIG_RECALL_2mm.nii.gz
#
# 3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz \
# -inset /home/kahwang/LESYMAP_for_Kai/CONS_CFT_RAW.nii.gz -prefix /home/kahwang/LESYMAP_for_Kai/CONS_CFT_2mm.nii.gz
#
# 3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz \
# -inset /home/kahwang/LESYMAP_for_Kai/COWA.nii.gz -prefix /home/kahwang/LESYMAP_for_Kai/COWA_2mm.nii.gz
#
# 3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz \
# -inset /home/kahwang/LESYMAP_for_Kai/TMTB.nii.gz -prefix /home/kahwang/LESYMAP_for_Kai/TMTB_2mm.nii.gz
#
#
# # separate Gray and White Matter
# 3dcalc -a /home/kahwang/LESYMAP_for_Kai/wm_2mm.nii -b /home/kahwang/LESYMAP_for_Kai/BNT_2mm.nii.gz \
# -expr 'step(a-0.5) * b' -prefix /home/kahwang/LESYMAP_for_Kai/BNT_WM.nii.gz
#
# 3dcalc -a /home/kahwang/LESYMAP_for_Kai/gm_2mm.nii -b /home/kahwang/LESYMAP_for_Kai/BNT_2mm.nii.gz \
# -expr 'step(a-0.5) * b' -prefix /home/kahwang/LESYMAP_for_Kai/BNT_GM.nii.gz
#
# 3dcalc -a /home/kahwang/LESYMAP_for_Kai/wm_2mm.nii -b /home/kahwang/LESYMAP_for_Kai/COM_FIG_RECALL_2mm.nii.gz \
# -expr 'step(a-0.5) * b' -prefix /home/kahwang/LESYMAP_for_Kai/COM_FIG_RECALL_WM.nii.gz
#
# 3dcalc -a /home/kahwang/LESYMAP_for_Kai/gm_2mm.nii -b /home/kahwang/LESYMAP_for_Kai/COM_FIG_RECALL_2mm.nii.gz \
# -expr 'step(a-0.5) * b' -prefix /home/kahwang/LESYMAP_for_Kai/COM_FIG_RECALL_GM.nii.gz
#
# 3dcalc -a /home/kahwang/LESYMAP_for_Kai/wm_2mm.nii -b /home/kahwang/LESYMAP_for_Kai/CONS_CFT_2mm.nii.gz \
# -expr 'step(a-0.5) * b' -prefix /home/kahwang/LESYMAP_for_Kai/CONS_CFT_WM.nii.gz
#
# 3dcalc -a /home/kahwang/LESYMAP_for_Kai/gm_2mm.nii -b /home/kahwang/LESYMAP_for_Kai/CONS_CFT_2mm.nii.gz \
# -expr 'step(a-0.5) * b' -prefix /home/kahwang/LESYMAP_for_Kai/CONS_CFT_GM.nii.gz
#
# 3dcalc -a /home/kahwang/LESYMAP_for_Kai/wm_2mm.nii -b /home/kahwang/LESYMAP_for_Kai/COWA_2mm.nii.gz \
# -expr 'step(a-0.5) * b' -prefix /home/kahwang/LESYMAP_for_Kai/COWA_WM.nii.gz
#
# 3dcalc -a /home/kahwang/LESYMAP_for_Kai/gm_2mm.nii -b /home/kahwang/LESYMAP_for_Kai/COWA_2mm.nii.gz \
# -expr 'step(a-0.5) * b' -prefix /home/kahwang/LESYMAP_for_Kai/COWA_GM.nii.gz
#
# 3dcalc -a /home/kahwang/LESYMAP_for_Kai/wm_2mm.nii -b /home/kahwang/LESYMAP_for_Kai/TMTB_2mm.nii.gz \
# -expr 'step(a-0.5) * b' -prefix /home/kahwang/LESYMAP_for_Kai/TMTB_WM.nii.gz
#
# 3dcalc -a /home/kahwang/LESYMAP_for_Kai/gm_2mm.nii -b /home/kahwang/LESYMAP_for_Kai/TMTB_2mm.nii.gz \
# -expr 'step(a-0.5) * b' -prefix /home/kahwang/LESYMAP_for_Kai/TMTB_GM.nii.gz

# Then use AFNI gui to number and label the GM clusters

# Then do FC using these cortical lesion clusters
# BNT_GM_Clust1.nii.gz  BNT_GM_Clust4.nii.gz          COM_FIG_RECALL_Clust3.nii.gz  COWA_Clust2.nii.gz
# BNT_GM_Clust2.nii.gz  COM_FIG_RECALL_Clust1.nii.gz  COM_FIG_RECALL_Clust4.nii.gz  TMTB_Clust1.nii.gz
# BNT_GM_Clust3.nii.gz  COM_FIG_RECALL_Clust2.nii.gz  COWA_Clust1.nii.gz            TMTB_Clust2.nii.gz

for subject in $(cat /data/backed_up/shared/Tha_Lesion_Mapping/Subject_List.txt); do

  # cortical lesion masks
	for mask in TMTB_Clust1 TMTB_Clust2; do
	 #BNT_GM_Clust1 BNT_GM_Clust2 BNT_GM_Clust3 BNT_GM_Clust4 COM_FIG_RECALL_Clust1 COM_FIG_RECALL_Clust2 COM_FIG_RECALL_Clust3 COM_FIG_RECALL_Clust4 COWA_Clust1 COWA_Clust2 TMTB_Clust1 TMTB_Clust2
	  rm -rf /data/backed_up/shared/Tha_Lesion_Mapping/${subject}/seed_corr_${mask}_ncsreg*
		3dNetCorr -inset /data/backed_up/shared/NKI/${subject}/MNINonLinear/rfMRI_REST_mx_1400_ncsreg.nii.gz \
		-in_rois /home/kahwang/LESYMAP_for_Kai/${mask}.nii.gz \
		-nifti \
		-ts_wb_Z \
		-prefix /data/backed_up/shared/Tha_Lesion_Mapping/${subject}/seed_corr_${mask}_ncsreg

	done
done


# group stats on lesion network
for mask in TMTB_Clust1 TMTB_Clust2; do

  rm /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_${mask}_ncsreg.nii.gz

	3dttest++ \
	-setA "/data/backed_up/shared/Tha_Lesion_Mapping/0*/seed_corr_${mask}_ncsreg_000_INDIV/WB_Z_ROI_00*.nii.gz" \
	-prefix /data/backed_up/shared/Tha_Lesion_Mapping/NKI_groupFC_${mask}_ncsreg.nii.gz

done


# MGH
for subject in $(cat /data/backed_up/shared/Tha_Lesion_Mapping/MGH_List_Bridged.txt); do

	3dTcat -prefix /data/backed_up/shared/MGH/MGH/${subject}/MNINonLinear/rfMRI_REST_ncsreg.nii.gz /data/backed_up/shared/MGH/MGH/${subject}/MNINonLinear/rfMRI_REST1_ncsreg.nii.gz /data/backed_up/shared/MGH/MGH/${subject}/MNINonLinear/rfMRI_REST2_ncsreg.nii.gz

	for mask in TMTB_Clust1 TMTB_Clust2; do
		rm -rf /data/backed_up/shared/Tha_Lesion_Mapping/MGH_${subject}/seed_corr_${mask}_ncsreg*
		3dNetCorr -inset /data/backed_up/shared/MGH/MGH/${subject}/MNINonLinear/rfMRI_REST_ncsreg.nii.gz \
		-in_rois /home/kahwang/LESYMAP_for_Kai/${mask}.nii.gz \
		-nifti \
		-ts_wb_Z \
		-prefix /data/backed_up/shared/Tha_Lesion_Mapping/MGH_${subject}/seed_corr_${mask}_ncsreg

	done
done


#### group stats on lesion network
for mask in TMTB_Clust1 TMTB_Clust2; do

  rm /data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_${mask}_ncsreg.nii.gz

	3dttest++ \
	-setA "/data/backed_up/shared/Tha_Lesion_Mapping/MGH*/seed_corr_${mask}_ncsreg_000_INDIV/WB_Z_ROI_00*.nii.gz" \
	-prefix /data/backed_up/shared/Tha_Lesion_Mapping/MGH_groupFC_${mask}_ncsreg.nii.gz

done
