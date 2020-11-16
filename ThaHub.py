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
# Run lesion network mapping, using lesion masks as seeds
########################################################################
# run Map_Network.sh