#Test multidomain hub properties in the human thalamus

import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from nibabel.processing import resample_from_to
import nilearn
import scipy
import os
import glob
import statsmodels.api as sm
import statsmodels.formula.api as smf
from nilearn import masking


# load data
df = pd.read_csv('~/RDSS/tmp/data.csv')

# remove acute patietns (chronicity < 3 months)
df = df.loc[df['Chronicity']>2].reset_index()


########################################################################
# normalize neuropsych data to popluation norm
########################################################################

# Norms for TMT
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

# compare neuropsych data
#BNT norms:
# 18-39 55.8 (3.8)
# 40-49 56.8 (3)
# 50-59 55.2 (4)
# 60-69 53.3 (4.6)
# 70-79 48.9 (6.3)

BNT_norm ={
'39': {'mean': 55.8, 'sd': 3.8},
'49': {'mean': 56.8 , 'sd': 3},
'59': {'mean': 55.2, 'sd': 4},
'69': {'mean': 53.3, 'sd': 4.6},
'79': {'mean': 48.9 , 'sd': 6.3},
}

#COWA norms: In years of educ
#M, 12  36.9 (9.8)
#   13-15   40.5 (9.4)
#   16  41 (9.8)
#F, 12  35.9 (9.6)
#   13-15   39.4 (10.1)
#   16  46.5(11.2)

COWA_norm = {
'M12': {'mean': 36.9, 'sd': 9.8},
'M15': {'mean': 40.5 , 'sd': 9.4},
'M16': {'mean': 41, 'sd': 9.8},
'F12': {'mean': 35.9, 'sd': 9.6},
'F15': {'mean': 39.4 , 'sd': 10.1},
'F16': {'mean': 46.5 , 'sd': 11.2},
}


# RVLT Delay recall
#   55-59   10.4 (3.1)
#   60-64   9.9 (3.1)
#   65-69   8.3 (3.5)
#   70-74   7.4 (3.1)
#   75-79   6.9 (2.9)
#   80-84   5.5 (3.3)
#   85-     5.4 (2.7)

RAVLT_Delayed_Recall_norm = {
'59': {'mean': 10.4, 'sd': 3.1},
'64': {'mean': 9.9 , 'sd': 3.1},
'69': {'mean': 8.3, 'sd': 3.5},
'74': {'mean': 7.4, 'sd': 3.1},
'79': {'mean': 6.9 , 'sd': 2.9},
'84': {'mean': 5.5 , 'sd': 3.3},
'85': {'mean': 5.4 , 'sd': 2.7},
}

# RAVLT  Recognition
#   55-59   14.1 (1.3)
#   60-64   13.9 (1.5)
#   65-69   13.3 (2)
#   70-74   12.7 (2.1)
#   75-79   12.5 (2.4)
#   80-84   12.3 (2.4)
#   85-     12.3 (2.3)

RAVLT_Recognition_norm = {
'59': {'mean': 14.1, 'sd': 1.3},
'64': {'mean': 13.9 , 'sd': 1.5},
'69': {'mean': 13.3, 'sd': 2},
'74': {'mean': 12.7, 'sd': 2.1},
'79': {'mean': 12.5 , 'sd': 2.4},
'84': {'mean': 12.3 , 'sd': 2.4},
'85': {'mean': 12.3 , 'sd': 2.3},
}

# rey o complex figure construction
#   40  32.83 (3.1)
#   50  31.79 (4.55)
#   60  31.94 (3.37)
#   70  31.76 (3.64)
#   80  30.14 (4.52)

Complex_Figure_Copy_norm = {
'40': {'mean': 32.83, 'sd': 3.1},
'50': {'mean': 31.79 , 'sd': 4.55},
'60': {'mean': 31.94, 'sd': 3.37},
'70': {'mean': 31.76, 'sd': 3.64},
'80': {'mean': 30.14 , 'sd': 4.52},
}

# rey o complex figure delayed recall
#   40  19.28 (7.29)
#   50  17.13 (7.14)
#   60  16.55 (6.08)
#   70  15.18 (5.57)
#   80  13.17 (5.32)
Complex_Figure_Recall_norm = {
'40': {'mean': 19.28, 'sd': 7.29},
'50': {'mean': 17.13 , 'sd': 7.14},
'60': {'mean': 16.55, 'sd': 6.08},
'70': {'mean': 15.18, 'sd': 5.57},
'80': {'mean': 13.17 , 'sd': 5.32},
}

# Convert raw scores to norm adjusted z score
for i in df.index:

	#Normalize TMT
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

	#Normalize BNT
	if df.loc[i, 'Age'] <= 39:
		df.loc[i, 'BNT_z'] = (df.loc[i, 'Boston'] - BNT_norm['39']['mean']) / BNT_norm['39']['sd']
	elif 39 < df.loc[i, 'Age'] <= 49:
		df.loc[i, 'BNT_z'] = (df.loc[i, 'Boston'] - BNT_norm['49']['mean']) / BNT_norm['49']['sd']
	elif 49 < df.loc[i, 'Age'] <= 59:
		df.loc[i, 'BNT_z'] = (df.loc[i, 'Boston'] - BNT_norm['59']['mean']) / BNT_norm['59']['sd']
	elif 59 < df.loc[i, 'Age'] <= 69:
		df.loc[i, 'BNT_z'] = (df.loc[i, 'Boston'] - BNT_norm['69']['mean']) / BNT_norm['69']['sd']
	elif 69 < df.loc[i, 'Age']:
		df.loc[i, 'BNT_z'] = (df.loc[i, 'Boston'] - BNT_norm['79']['mean']) / BNT_norm['79']['sd']

	#Cowa
	if (df.loc[i, 'Sex'] == 'M') & (df.loc[i, 'Educ'] <= 12):
		df.loc[i, 'COWA_z'] = (df.loc[i, 'COWA'] - COWA_norm['M12']['mean']) / COWA_norm['M12']['sd']
	elif (df.loc[i, 'Sex'] == 'M') & (12 < df.loc[i, 'Educ'] <= 15):
		df.loc[i, 'COWA_z'] = (df.loc[i, 'COWA'] - COWA_norm['M15']['mean']) / COWA_norm['M15']['sd']
	elif (df.loc[i, 'Sex'] == 'M') & (15 < df.loc[i, 'Educ']):
		df.loc[i, 'COWA_z'] = (df.loc[i, 'COWA'] - COWA_norm['M16']['mean']) / COWA_norm['M16']['sd']
	elif (df.loc[i, 'Sex'] == 'F') & (df.loc[i, 'Educ'] <= 12):
		df.loc[i, 'COWA_z'] = (df.loc[i, 'COWA'] - COWA_norm['F12']['mean']) / COWA_norm['F12']['sd']
	elif (df.loc[i, 'Sex'] == 'F') & (12 < df.loc[i, 'Educ'] <= 15):
		df.loc[i, 'COWA_z'] = (df.loc[i, 'COWA'] - COWA_norm['F15']['mean']) / COWA_norm['F15']['sd']
	elif (df.loc[i, 'Sex'] == 'F') & (15 < df.loc[i, 'Educ']):
		df.loc[i, 'COWA_z'] = (df.loc[i, 'COWA'] - COWA_norm['F16']['mean']) / COWA_norm['F16']['sd']

	#RAVLT
	if df.loc[i, 'Age'] <= 59:
		df.loc[i, 'RAVLT_Delayed_Recall_z'] = (df.loc[i, 'RAVLT_Delayed_Recall'] - RAVLT_Delayed_Recall_norm['59']['mean']) / RAVLT_Delayed_Recall_norm['59']['sd']
	elif 59 < df.loc[i, 'Age'] <= 64:
		df.loc[i, 'RAVLT_Delayed_Recall_z'] = (df.loc[i, 'RAVLT_Delayed_Recall'] - RAVLT_Delayed_Recall_norm['64']['mean']) / RAVLT_Delayed_Recall_norm['64']['sd']
	elif 64 < df.loc[i, 'Age'] <= 69:
		df.loc[i, 'RAVLT_Delayed_Recall_z'] = (df.loc[i, 'RAVLT_Delayed_Recall'] - RAVLT_Delayed_Recall_norm['69']['mean']) / RAVLT_Delayed_Recall_norm['69']['sd']
	elif 69 < df.loc[i, 'Age'] <= 74:
		df.loc[i, 'RAVLT_Delayed_Recall_z'] = (df.loc[i, 'RAVLT_Delayed_Recall'] - RAVLT_Delayed_Recall_norm['74']['mean']) / RAVLT_Delayed_Recall_norm['74']['sd']
	elif 74 < df.loc[i, 'Age'] <= 79:
		df.loc[i, 'RAVLT_Delayed_Recall_z'] = (df.loc[i, 'RAVLT_Delayed_Recall'] - RAVLT_Delayed_Recall_norm['79']['mean']) / RAVLT_Delayed_Recall_norm['79']['sd']
	elif 79 < df.loc[i, 'Age'] <= 84:
		df.loc[i, 'RAVLT_Delayed_Recall_z'] = (df.loc[i, 'RAVLT_Delayed_Recall'] - RAVLT_Delayed_Recall_norm['84']['mean']) / RAVLT_Delayed_Recall_norm['84']['sd']
	elif 84 < df.loc[i, 'Age']:
		df.loc[i, 'RAVLT_Delayed_Recall_z'] = (df.loc[i, 'RAVLT_Delayed_Recall'] - RAVLT_Delayed_Recall_norm['85']['mean']) / RAVLT_Delayed_Recall_norm['85']['sd']

	if df.loc[i, 'Age'] <= 59:
		df.loc[i, 'RAVLT_Recognition_z'] = (df.loc[i, 'RAVLT_Hit'] - RAVLT_Recognition_norm['59']['mean']) / RAVLT_Recognition_norm['59']['sd']
	elif 59 < df.loc[i, 'Age'] <= 64:
		df.loc[i, 'RAVLT_Recognition_z'] = (df.loc[i, 'RAVLT_Hit'] - RAVLT_Recognition_norm['64']['mean']) / RAVLT_Recognition_norm['64']['sd']
	elif 64 < df.loc[i, 'Age'] <= 69:
		df.loc[i, 'RAVLT_Recognition_z'] = (df.loc[i, 'RAVLT_Hit'] - RAVLT_Recognition_norm['69']['mean']) / RAVLT_Recognition_norm['69']['sd']
	elif 69 < df.loc[i, 'Age'] <= 74:
		df.loc[i, 'RAVLT_Recognition_z'] = (df.loc[i, 'RAVLT_Hit'] - RAVLT_Recognition_norm['74']['mean']) / RAVLT_Recognition_norm['74']['sd']
	elif 74 < df.loc[i, 'Age'] <= 79:
		df.loc[i, 'RAVLT_Recognition_z'] = (df.loc[i, 'RAVLT_Hit'] - RAVLT_Recognition_norm['79']['mean']) / RAVLT_Recognition_norm['79']['sd']
	elif 79 < df.loc[i, 'Age'] <= 84:
		df.loc[i, 'RAVLT_Recognition_z'] = (df.loc[i, 'RAVLT_Hit'] - RAVLT_Recognition_norm['84']['mean']) / RAVLT_Recognition_norm['84']['sd']
	elif 84 < df.loc[i, 'Age']:
		df.loc[i, 'RAVLT_Recognition_z'] = (df.loc[i, 'RAVLT_Hit'] - RAVLT_Recognition_norm['85']['mean']) / RAVLT_Recognition_norm['85']['sd']

	if df.loc[i, 'Age'] <= 40:
		df.loc[i, 'Complex_Figure_Copy_z'] = (df.loc[i, 'Complex_Figure_Copy'] - Complex_Figure_Copy_norm['40']['mean']) / Complex_Figure_Copy_norm['40']['sd']
	elif 40 < df.loc[i, 'Age'] <= 50:
		df.loc[i, 'Complex_Figure_Copy_z'] = (df.loc[i, 'Complex_Figure_Copy'] - Complex_Figure_Copy_norm['50']['mean']) / Complex_Figure_Copy_norm['50']['sd']
	elif 50 < df.loc[i, 'Age'] <= 60:
		df.loc[i, 'Complex_Figure_Copy_z'] = (df.loc[i, 'Complex_Figure_Copy'] - Complex_Figure_Copy_norm['60']['mean']) / Complex_Figure_Copy_norm['60']['sd']
	elif 60 < df.loc[i, 'Age'] <= 70:
		df.loc[i, 'Complex_Figure_Copy_z'] = (df.loc[i, 'Complex_Figure_Copy'] - Complex_Figure_Copy_norm['70']['mean']) / Complex_Figure_Copy_norm['70']['sd']
	elif 70 < df.loc[i, 'Age']:
		df.loc[i, 'Complex_Figure_Copy_z'] = (df.loc[i, 'Complex_Figure_Copy'] - Complex_Figure_Copy_norm['80']['mean']) / Complex_Figure_Copy_norm['80']['sd']

	if df.loc[i, 'Age'] <= 40:
		df.loc[i, 'Complex_Figure_Recall_z'] = (df.loc[i, 'Complex_Figure_Recall'] - Complex_Figure_Recall_norm['40']['mean']) / Complex_Figure_Recall_norm['40']['sd']
	elif 40 < df.loc[i, 'Age'] <= 50:
		df.loc[i, 'Complex_Figure_Recall_z'] = (df.loc[i, 'Complex_Figure_Recall'] - Complex_Figure_Recall_norm['50']['mean']) / Complex_Figure_Recall_norm['50']['sd']
	elif 50 < df.loc[i, 'Age'] <= 60:
		df.loc[i, 'Complex_Figure_Recall_z'] = (df.loc[i, 'Complex_Figure_Recall'] - Complex_Figure_Recall_norm['60']['mean']) / Complex_Figure_Recall_norm['60']['sd']
	elif 60 < df.loc[i, 'Age'] <= 70:
		df.loc[i, 'Complex_Figure_Recall_z'] = (df.loc[i, 'Complex_Figure_Recall'] - Complex_Figure_Recall_norm['70']['mean']) / Complex_Figure_Recall_norm['70']['sd']
	elif 70 < df.loc[i, 'Age']:
		df.loc[i, 'Complex_Figure_Recall_z'] = (df.loc[i, 'Complex_Figure_Recall'] - Complex_Figure_Recall_norm['80']['mean']) / Complex_Figure_Recall_norm['80']['sd']

df.to_csv('~/RDSS/tmp/data_z.csv')



########################################################################
# Calculate lesion size
########################################################################

for i in df.index:
	# load mask and get size
	s = df.loc[i, 'Sub']

	# annoying zeros
	if s == '902':
		s = '0902'
	if s == '802':
		s = '0802'
	try:
		fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
		m = nib.load(fn)
		df.loc[i, 'Lesion Size'] = np.sum(m.get_data()) * 0.125 #0.5 isotropic voxels, cubic = 0.125
	except:
		df.loc[i, 'Lesion Size'] = np.nan

df.to_csv('~/RDSS/tmp/data_z.csv')



########################################################################
# Check comparison pateitns lesion overlap with LESYMAP results, select appropriate controls
########################################################################

# load Lesymap lesion sympton maps from various tasks
TMTB_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/Trail_making_part_B_LESYMAP/stat_img.nii.gz')
TMTB_LESYMAP_map = TMTB_LESYMAP.get_data()
BNT_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/BOS_NAM_RAW/stat_img.nii.gz')
BNT_LESYMAP_map = BNT_LESYMAP.get_data()
COWA_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/MAE_COWA/stat_img.nii.gz')
COWA_LESYMAP_map = COWA_LESYMAP.get_data()
COM_FIG_COPY_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/CONS_CFT_RAW/stat_img.nii.gz')
COM_FIG_COPY_LESYMAP_map = COM_FIG_COPY_LESYMAP.get_data()
COM_FIG_RECALL_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/COM_FIG_RECALL/stat_img.nii.gz')
COM_FIG_RECALL_LESYMAP_map = COM_FIG_RECALL_LESYMAP.get_data()


for i in df.index:
	# load mask and get size
	s = df.loc[i, 'Sub']

	# annoying zeros
	if s == '902':
		s = '0902'
	if s == '802':
		s = '0802'
	try:
		fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
		m = nib.load(fn)
		res_m = resample_from_to(m, TMTB_LESYMAP).get_data()

		if np.sum(res_m*TMTB_LESYMAP_map)==0:
			df.loc[i, 'TMTB_Comparison'] = True
		else:
			df.loc[i, 'TMTB_Comparison'] = False

		if np.sum(res_m*BNT_LESYMAP_map)==0:
			df.loc[i, 'BNT_Comparison'] = True
		else:
			df.loc[i, 'BNT_Comparison'] = False

		if np.sum(res_m*COWA_LESYMAP_map)==0:
			df.loc[i, 'COWA_Comparison'] = True
		else:
			df.loc[i, 'COWA_Comparison'] = False

		if np.sum(res_m*COM_FIG_COPY_LESYMAP_map)==0:
			df.loc[i, 'Complex_Figure_Copy_Comparison'] = True
		else:
			df.loc[i, 'Complex_Figure_Copy_Comparison'] = False

		if np.sum(res_m*COM_FIG_RECALL_LESYMAP_map)==0:
			df.loc[i, 'Complex_Figure_Recall_Comparison'] = True
		else:
			df.loc[i, 'Complex_Figure_Recall_Comparison'] = False

	except:
		continue

df.to_csv('~/RDSS/tmp/data_z.csv')


########################################################################
# Compare test scores between patient groups
########################################################################

df = pd.read_csv('~/RDSS/tmp/data_z.csv')

scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx') & (df['TMTB_Comparison']==True)]['TMTA_z'].values, df.loc[df['Site']=='Th']['TMTA_z'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx') & (df['TMTB_Comparison']==True)]['TMTB_z'].values, df.loc[df['Site']=='Th']['TMTB_z'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx') & (df['BNT_Comparison']==True)]['BNT_z'].values, df.loc[df['Site']=='Th']['BNT_z'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx') & (df['COWA_Comparison']==True)]['COWA_z'].values, df.loc[df['Site']=='Th']['COWA_z'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['RAVLT_Delayed_Recall_z'].values, df.loc[df['Site']=='Th']['RAVLT_Delayed_Recall_z'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['RAVLT_Recognition_z'].values, df.loc[df['Site']=='Th']['RAVLT_Recognition_z'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx') & (df['Complex_Figure_Copy_Comparison']==True)]['Complex_Figure_Copy_z'].values, df.loc[df['Site']=='Th']['Complex_Figure_Copy_z'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx') & (df['Complex_Figure_Recall_Comparison']==True)]['Complex_Figure_Recall_z'].values, df.loc[df['Site']=='Th']['Complex_Figure_Recall_z'].values)


########################################################################
# Create table of impairment
########################################################################

df['TMTB_z_Impaired'] = df['TMTB_z'] >1
df['BNT_z_Impaired'] = df['BNT_z'] <-1
df['COWA_z_Impaired'] = df['COWA_z'] <-1
df['RAVLT_Delayed_Recall_z_Impaired'] = df['RAVLT_Delayed_Recall_z'] <-1
df['RAVLT_Recognition_z_Impaired'] = df['RAVLT_Recognition_z'] <-1
df['Complex_Figure_Copy_z_Impaired'] = df['Complex_Figure_Copy_z'] <-1
df['Complex_Figure_Recall_z_Impaired'] = df['Complex_Figure_Recall_z'] <-1

df['MM_impaired'] = sum([df['TMTB_z_Impaired'] , df['BNT_z_Impaired'] , df['COWA_z_Impaired'] , df['RAVLT_Delayed_Recall_z_Impaired'] , df['RAVLT_Recognition_z_Impaired'] , df['Complex_Figure_Copy_z_Impaired'] , df['Complex_Figure_Recall_z_Impaired']])

df.to_csv('~/RDSS/tmp/data_z.csv')

########################################################################
# Run lesion network mapping
########################################################################
### run Map_Network.sh, this one use thalamus lesion mask at seed and check overlap with cortical LESYMAP masks

#resample LESYMAP output from 1mm grid to 2mm grid (FC maps in 2mm grid)
# os.system("3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz -inset /home/kahwang/LESYMAP_for_Kai/Trail_making_part_B_LESYMAP/stat_img.nii.gz -prefix /home/kahwang/LESYMAP_for_Kai/Trail_making_part_B_LESYMAP/stat_img_2mm.nii.gz")
# os.system("3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz -inset /home/kahwang/LESYMAP_for_Kai/BOS_NAM_RAW/stat_img.nii.gz -prefix /home/kahwang/LESYMAP_for_Kai/BOS_NAM_RAW/stat_img_2mm.nii.gz")
# os.system("3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz -inset /home/kahwang/LESYMAP_for_Kai/MAE_COWA/stat_img.nii.gz -prefix /home/kahwang/LESYMAP_for_Kai/MAE_COWA/stat_img_2mm.nii.gz")
# os.system("3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz -inset /home/kahwang/LESYMAP_for_Kai/CONS_CFT_RAW/stat_img.nii.gz -prefix /home/kahwang/LESYMAP_for_Kai/CONS_CFT_RAW/stat_img_2mm.nii.gz")
# os.system("3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz -inset /home/kahwang/LESYMAP_for_Kai/COM_FIG_RECALL/stat_img.nii.gz -prefix /home/kahwang/LESYMAP_for_Kai/COM_FIG_RECALL/stat_img_2mm.nii.gz")
# os.system("3dresample -master /data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz -inset /home/kahwang/bsh/standard/mni_icbm152_nlin_asym_09c/mni_icbm152_wm_tal_nlin_asym_09c_2mm.nii -prefix /home/kahwang/bsh/standard/mni_icbm152_nlin_asym_09c/wm_2mm.nii")

TMTB_LESYMAP_map = nib.load('/home/kahwang/LESYMAP_for_Kai/Trail_making_part_B_LESYMAP/stat_img_2mm.nii.gz').get_data()!=0
BNT_LESYMAP_map = nib.load('/home/kahwang/LESYMAP_for_Kai/BOS_NAM_RAW/stat_img_2mm.nii.gz').get_data()!=0
COWA_LESYMAP_map = nib.load('/home/kahwang/LESYMAP_for_Kai/MAE_COWA/stat_img_2mm.nii.gz').get_data()!=0
COM_FIG_COPY_LESYMAP_map = nib.load('/home/kahwang/LESYMAP_for_Kai/CONS_CFT_RAW/stat_img_2mm.nii.gz').get_data()!=0
COM_FIG_RECALL_LESYMAP_map = nib.load('/home/kahwang/LESYMAP_for_Kai/COM_FIG_RECALL/stat_img_2mm.nii.gz').get_data()!=0

# use MNI 2mm template as base
m = nib.load('/data/backed_up/shared/Tha_Lesion_Mapping/MNI_brainamsk_2mm.nii.gz')
#load WM atlas
WM_atlas = nib.load('/home/kahwang/bsh/standard/mni_icbm152_nlin_asym_09c/wm_2mm.nii').get_data() > 0.5
#WM_atlas = WM_atlas >0.5

# TMTB_LESYMAP_map = resample_from_to(TMTB_LESYMAP, m).get_data()>0
# BNT_LESYMAP_map = resample_from_to(BNT_LESYMAP, m).get_data()>0
# COWA_LESYMAP_map = resample_from_to(COWA_LESYMAP, m).get_data()>0
# COM_FIG_COPY_LESYMAP_map = resample_from_to(COM_FIG_COPY_LESYMAP, m).get_data()>0
# COM_FIG_RECALL_LESYMAP_map = resample_from_to(COM_FIG_RECALL_LESYMAP, m).get_data()>0
# from nilearn import plotting
# plotting.plot_glass_brain(resample_from_to(TMTB_LESYMAP, m), threshold=1)


# get rid of WM
TMTB_LESYMAP_map = 1* ((1 * TMTB_LESYMAP_map - 1 * WM_atlas) > 0)
BNT_LESYMAP_map = 1* ((1 * BNT_LESYMAP_map - 1 * WM_atlas) > 0)
COWA_LESYMAP_map = 1* ((1 * COWA_LESYMAP_map - 1 * WM_atlas) > 0)
COM_FIG_COPY_LESYMAP_map = 1* ((1 * COM_FIG_COPY_LESYMAP_map - 1 * WM_atlas) > 0)
COM_FIG_RECALL_LESYMAP_map = 1* ((1 * COM_FIG_RECALL_LESYMAP_map - 1 * WM_atlas) > 0)
#
# a = nilearn.image.new_img_like(m, TMTB_LESYMAP_map, copy_header=True)
# a.to_filename('test.nii')
#plotting.plot_glass_brain(resample_from_to(TMTB_LESYMAP, m), threshold=0.8)
#plotting.show()

for p in df.loc[df['Site'] == 'Th']['Sub']:

	if p == '4045':
		continue #no mask yet
	else:
		fcfile = '/home/kahwang/bsh/Tha_Lesion_Mapping/MGH_groupFC_%s.nii.gz'  %p
		fcmap = nib.load(fcfile).get_data()[:,:,:,0,1]

		df.loc[df['Sub'] == p, 'TMTB_FC'] = np.mean(fcmap[TMTB_LESYMAP_map >0])
		df.loc[df['Sub'] == p, 'BNT_FC'] = np.mean(fcmap[BNT_LESYMAP_map >0])
		df.loc[df['Sub'] == p, 'COWA_FC'] = np.mean(fcmap[COWA_LESYMAP_map >0])
		df.loc[df['Sub'] == p, 'COM_FIG_COPY_FC'] = np.mean(fcmap[COM_FIG_COPY_LESYMAP_map >0])
		df.loc[df['Sub'] == p, 'COM_FIG_RECALL_FC'] = np.mean(fcmap[COM_FIG_RECALL_LESYMAP_map >0])

print(df.groupby(['TMTB_z_Impaired'])['TMTB_FC'].mean())
print(df.groupby(['BNT_z_Impaired'])['BNT_FC'].mean())
print(df.groupby(['COWA_z_Impaired'])['COWA_FC'].mean())
print(df.groupby(['Complex_Figure_Copy_z_Impaired'])['COM_FIG_COPY_FC'].mean())
print(df.groupby(['Complex_Figure_Recall_z_Impaired'])['COM_FIG_RECALL_FC'].mean())

scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['TMTB_z_Impaired']==True)]['TMTB_FC'].values, df.loc[(df['Site']=='Th') & (df['TMTB_z_Impaired']==False)]['TMTB_FC'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==True)]['BNT_FC'].values, df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==False)]['BNT_FC'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['COWA_z_Impaired']==True)]['COWA_FC'].values, df.loc[(df['Site']=='Th') & (df['COWA_z_Impaired']==False)]['COWA_FC'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['Complex_Figure_Copy_z_Impaired']==True)]['COM_FIG_COPY_FC'].values, df.loc[(df['Site']=='Th') & (df['Complex_Figure_Copy_z_Impaired']==False)]['COM_FIG_COPY_FC'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==True)]['COM_FIG_RECALL_FC'].values, df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==False)]['COM_FIG_RECALL_FC'].values)



### run Prep_LESYMAP_clusters.sh, this one use cortical lesion masks, separated by GM, as seeds
lesymap_clusters = ['BNT_GM_Clust1', 'BNT_GM_Clust2', 'BNT_GM_Clust3', 'BNT_GM_Clust4', 'COM_FIG_RECALL_Clust1', 'COM_FIG_RECALL_Clust2', 'COM_FIG_RECALL_Clust3', 'COM_FIG_RECALL_Clust4', 'COWA_Clust1', 'COWA_Clust2', 'TMTB_Clust1', 'TMTB_Clust2']

for lesymap in lesymap_clusters:
	fcfile = '/home/kahwang/bsh/Tha_Lesion_Mapping/MGH_groupFC_%s_ncsreg.nii.gz'  %lesymap
	fcmap = nib.load(fcfile).get_data()[:,:,:,0,1]

	for p in df.loc[df['Site'] == 'Th']['Sub']:
		try:
			fn = '/home/kahwang/0.5mm/%s_2mm.nii.gz' %p
			m = nib.load(fn).get_data()
		except:
			continue

		df.loc[df['Sub'] == p, lesymap] = np.mean(fcmap[m>0])

print(df.groupby(['TMTB_z_Impaired'])['TMTB_Clust1'].mean())
print(df.groupby(['TMTB_z_Impaired'])['TMTB_Clust2'].mean())
print(df.groupby(['BNT_z_Impaired'])['BNT_GM_Clust1'].mean())
print(df.groupby(['BNT_z_Impaired'])['BNT_GM_Clust2'].mean())
print(df.groupby(['BNT_z_Impaired'])['BNT_GM_Clust3'].mean())
print(df.groupby(['BNT_z_Impaired'])['BNT_GM_Clust4'].mean())
print(df.groupby(['COWA_z_Impaired'])['COWA_Clust1'].mean())
print(df.groupby(['COWA_z_Impaired'])['COWA_Clust2'].mean())
print(df.groupby(['Complex_Figure_Recall_z_Impaired'])['COM_FIG_RECALL_Clust1'].mean())
print(df.groupby(['Complex_Figure_Recall_z_Impaired'])['COM_FIG_RECALL_Clust2'].mean())
print(df.groupby(['Complex_Figure_Recall_z_Impaired'])['COM_FIG_RECALL_Clust3'].mean())
print(df.groupby(['Complex_Figure_Recall_z_Impaired'])['COM_FIG_RECALL_Clust4'].mean())

scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['TMTB_z_Impaired']==True)]['TMTB_Clust1'].values, df.loc[(df['Site']=='Th') & (df['TMTB_z_Impaired']==False)]['TMTB_Clust1'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['TMTB_z_Impaired']==True)]['TMTB_Clust2'].values, df.loc[(df['Site']=='Th') & (df['TMTB_z_Impaired']==False)]['TMTB_Clust2'].values)

scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==True)]['BNT_GM_Clust1'].values, df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==False)]['BNT_GM_Clust1'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==True)]['BNT_GM_Clust2'].values, df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==False)]['BNT_GM_Clust2'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==True)]['BNT_GM_Clust3'].values, df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==False)]['BNT_GM_Clust3'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==True)]['BNT_GM_Clust4'].values, df.loc[(df['Site']=='Th') & (df['BNT_z_Impaired']==False)]['BNT_GM_Clust4'].values)

scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==True)]['COM_FIG_RECALL_Clust1'].values, df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==False)]['COM_FIG_RECALL_Clust1'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==True)]['COM_FIG_RECALL_Clust2'].values, df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==False)]['COM_FIG_RECALL_Clust2'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==True)]['COM_FIG_RECALL_Clust3'].values, df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==False)]['COM_FIG_RECALL_Clust3'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==True)]['COM_FIG_RECALL_Clust4'].values, df.loc[(df['Site']=='Th') & (df['Complex_Figure_Recall_z_Impaired']==False)]['COM_FIG_RECALL_Clust4'].values)


##### Use linear mixed effect regression model to test FC differences
lesymap_clusters = ['BNT_GM_Clust1', 'BNT_GM_Clust2', 'BNT_GM_Clust3', 'BNT_GM_Clust4', 'COM_FIG_RECALL_Clust1', 'COM_FIG_RECALL_Clust2', 'COM_FIG_RECALL_Clust3', 'COM_FIG_RECALL_Clust4', 'COWA_Clust1', 'COWA_Clust2', 'TMTB_Clust1', 'TMTB_Clust2']
fcdf = pd.DataFrame()

i=0
for p in df.loc[df['Site'] == 'Th']['Sub']:
	try:
		fn = '/home/kahwang/0.5mm/%s_2mm.nii.gz' %p
		m = nib.load(fn).get_data()
	except:
		continue

	for lesymap in lesymap_clusters:
		fn = '/data/backed_up/shared/Tha_Lesion_Mapping/MGH*/seed_corr_%s_ncsreg_000_INDIV/*.nii.gz' %lesymap
		files = glob.glob(fn)

		s = 0
		for f in files:
			fcmap = nib.load(f).get_data()
			fcdf.loc[i, 'Subject'] = str(s)
			fcdf.loc[i, 'Patient'] = p
			fcdf.loc[i, 'Cluster'] = lesymap
			fcdf.loc[i, 'FC'] = np.mean(fcmap[m>0])
			fcdf.loc[i, 'TMTB_z_Impaired'] = df.loc[df['Sub'] == p]['TMTB_z_Impaired'].values[0]
			fcdf.loc[i, 'BNT_z_Impaired'] = df.loc[df['Sub'] == p]['BNT_z_Impaired'].values[0]
			fcdf.loc[i, 'COWA_z_Impaired'] = df.loc[df['Sub'] == p]['COWA_z_Impaired'].values[0]
			fcdf.loc[i, 'Complex_Figure_Recall_z_Impaired'] = df.loc[df['Sub'] == p]['Complex_Figure_Recall_z_Impaired'].values[0]

			if lesymap == 'BNT_GM_Clust1':
				fcdf.loc[i, 'Task'] = 'BNT'
			if lesymap == 'BNT_GM_Clust2':
				fcdf.loc[i, 'Task'] = 'BNT'
			if lesymap == 'BNT_GM_Clust3':
				fcdf.loc[i, 'Task'] = 'BNT'
			if lesymap == 'BNT_GM_Clust4':
				fcdf.loc[i, 'Task'] = 'BNT'
			if lesymap == 'COM_FIG_RECALL_Clust1':
				fcdf.loc[i, 'Task'] = 'COM_FIG_RECALL'
			if lesymap == 'COM_FIG_RECALL_Clust2':
				fcdf.loc[i, 'Task'] = 'COM_FIG_RECALL'
			if lesymap == 'COM_FIG_RECALL_Clust3':
				fcdf.loc[i, 'Task'] = 'COM_FIG_RECALL'
			if lesymap == 'COM_FIG_RECALL_Clust4':
				fcdf.loc[i, 'Task'] = 'COM_FIG_RECALL'
			if lesymap == 'COWA_Clust1':
				fcdf.loc[i, 'Task'] = 'COWA'
			if lesymap == 'COWA_Clust2':
				fcdf.loc[i, 'Task'] = 'COWA'
			if lesymap == 'TMTB_Clust1':
				fcdf.loc[i, 'Task'] = 'TMTB'
			if lesymap == 'TMTB_Clust2':
				fcdf.loc[i, 'Task'] = 'TMTB'

			i = i+1
			s = s+1

fcdf.to_csv('~/RDSS/tmp/fcdata.csv')

# statsmodel, subject with random intercept
# https://www.statsmodels.org/stable/mixed_linear.html,

tdf = fcdf.loc[fcdf['Task'] == 'TMTB']
md = smf.mixedlm("FC ~ Cluster + TMTB_z_Impaired", tdf, groups=tdf['Subject']).fit()
print(md.summary())

tdf = fcdf.loc[fcdf['Task'] == 'COWA']
md = smf.mixedlm("FC ~ Cluster + COWA_z_Impaired", tdf, groups=tdf['Subject']).fit()
print(md.summary())

tdf = fcdf.loc[fcdf['Task'] == 'COM_FIG_RECALL']
md = smf.mixedlm("FC ~ Cluster + Complex_Figure_Recall_z_Impaired", tdf, groups=tdf['Subject']).fit()
print(md.summary())

tdf = fcdf.loc[fcdf['Task'] == 'BNT']
md = smf.mixedlm("FC ~ Cluster + BNT_z_Impaired", tdf, groups=tdf['Subject']).fit()
print(md.summary())

########################################################################
# Calculate participation coef for each lesion mask,
# Determine nuclei overlap for each lesion mask
########################################################################

# PC
PC_map = cmap = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/FC_analysis/PC.nii.gz')

#nuclei
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

morel_atlas = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_data()

for p in df.loc[df['Site'] == 'Th']['Sub'] :

	if p == '4045':
		continue #no mask yet

	else:
		#resample mask from .5 mm grid to 2 mm grid
		cmd = "3dresample -master /data/backed_up/kahwang/Tha_Neuropsych/FC_analysis/PC.nii.gz -inset /home/kahwang/0.5mm/%s.nii.gz -prefix /home/kahwang/0.5mm/%s_2mm.nii.gz" %(p, p)
		os.system(cmd)

		fn = '/home/kahwang/0.5mm/%s_2mm.nii.gz' %p
		lesion_mask = nib.load(fn)

		df.loc[df['Sub']==p,'PC'] = np.mean(PC_map.get_data()[lesion_mask.get_data()>0])

		for n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]:
			df.loc[df['Sub']==p, Morel[n]] = 8 * np.sum(lesion_mask.get_data()[morel_atlas==n])

scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th') & (df['MM_impaired']>2)]['PC'].values, df.loc[(df['Site']=='Th') & (df['MM_impaired']<2)]['PC'].values)
scipy.stats.mannwhitneyu(df.loc[(df['Site']=='Th')]['MM_impaired'].values, df.loc[(df['Site']=='ctx')]['MM_impaired'].values)


df.to_csv('~/RDSS/tmp/data_z.csv')

########################################################################
# Calculate lesymap FC's participation coef, using different lesymap seed FC as "communities"
########################################################################

### Use nilearn masker to get every tha voxel
#Thalamus mask
thalamus_mask = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz')
thalamus_mask_data = nib.load('/data/backed_up/kahwang/Tha_Neuropsych/ROI/Thalamus_Morel_consolidated_mask_v3.nii.gz').get_data()
thalamus_mask_data = thalamus_mask_data>0
thalamus_mask = nilearn.image.new_img_like(thalamus_mask, thalamus_mask_data, copy_header = True)

fn = '/data/backed_up/shared/Tha_Lesion_Mapping/MGH*/seed_corr_BNT_GM_Clust1_ncsreg_000_INDIV/*.nii.gz'
files = glob.glob(fn)

# FCMaps
lesymap_clusters = ['BNT_GM_Clust1', 'BNT_GM_Clust2', 'BNT_GM_Clust3', 'BNT_GM_Clust4', 'COM_FIG_RECALL_Clust1', 'COM_FIG_RECALL_Clust2', 'COM_FIG_RECALL_Clust3', 'COM_FIG_RECALL_Clust4', 'COWA_Clust1', 'COWA_Clust2', 'TMTB_Clust1', 'TMTB_Clust2']
# How FC Maps are grouped into task indices
lesymap_CI = np.array([1, 1, 1 , 1, 2, 2, 2, 2, 3, 3, 4, 4])

fc_vectors = np.zeros((np.count_nonzero(thalamus_mask_data>0),len(lesymap_CI), len(files)))
for il, lesymap in enumerate(lesymap_clusters):
	fn = '/data/backed_up/shared/Tha_Lesion_Mapping/MGH*/seed_corr_%s_ncsreg_000_INDIV/*.nii.gz' %lesymap
	files = glob.glob(fn)
	for ix, f in enumerate(files):
		fcmap = nib.load(f)
		fc_vectors[:,il, ix] = masking.apply_mask(fcmap, thalamus_mask)


np.save('fc_vectors', fc_vectors)
fc_vectors = np.load('fc_vectors.npy')
fc_vectors[fc_vectors<0] = 0

#FC's PC calculation
fc_sum = np.sum(fc_vectors, axis = 1)

kis = np.zeros(np.shape(fc_sum))
for ci in np.array([1, 2, 3, 4]):
	kis = kis + np.square(np.sum(fc_vectors[:,np.where(lesymap_CI==ci)[0],:], axis=1) / fc_sum)
fc_pc = 1-kis


pcdf = pd.DataFrame()
i=0
for p in df.loc[df['Site'] == 'Th']['Sub']:
	try:
		fn = '/home/kahwang/0.5mm/%s_2mm.nii.gz' %p
		m = nib.load(fn).get_data()
	except:
		continue

	for s in np.arange(0, np.shape(fc_pc)[1]):
		pc = fc_pc[:,s]
		pc_image = masking.unmask(pc, thalamus_mask).get_data()

		pcdf.loc[i, 'Subject'] = str(s)
		pcdf.loc[i, 'Patient'] = p
		pcdf.loc[i, 'PC'] = np.mean(pc_image[np.where(m>0)][pc_image[np.where(m>0)]>0])
		pcdf.loc[i, 'TMTB_z_Impaired'] = df.loc[df['Sub'] == p]['TMTB_z_Impaired'].values[0]
		pcdf.loc[i, 'BNT_z_Impaired'] = df.loc[df['Sub'] == p]['BNT_z_Impaired'].values[0]
		pcdf.loc[i, 'COWA_z_Impaired'] = df.loc[df['Sub'] == p]['COWA_z_Impaired'].values[0]
		pcdf.loc[i, 'Complex_Figure_Recall_z_Impaired'] = df.loc[df['Sub'] == p]['Complex_Figure_Recall_z_Impaired'].values[0]
		pcdf.loc[i, 'MM_impaired_num'] = df.loc[df['Sub'] == p]['MM_impaired'].values[0]
		pcdf.loc[i, 'MM_impaired'] = df.loc[df['Sub'] == p]['MM_impaired'].values[0]>2
		i = i+1

pcdf['MM_impaired'] = pcdf['MM_impaired_num'] >=2
md = smf.mixedlm("PC ~ MM_impaired", pcdf, groups=pcdf['Subject']).fit()
print(md.summary())

md = smf.mixedlm("PC ~ MM_impaired_num", pcdf, groups=pcdf['Subject']).fit()
print(md.summary())

# from nilearn import plotting
# pc_image = masking.unmask(pc, thalamus_mask)
# plotting.plot_glass_brain(pc_image, threshold=0.01)
# plotting.show()
#
# plotting.plot_glass_brain(nib.load(fn), threshold=0.01)
# plotting.show()

#end of line
