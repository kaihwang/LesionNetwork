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
from nilearn import plotting
import matplotlib.pyplot as plt
sns.set_context("paper")
###########
########### Test multidomain hub properties in the human thalamus
###########

def load_and_normalize_neuropsych_data():
	''' normalize neuropsych data using popluation norm'''

	# load data
	df = pd.read_csv('~/RDSS/tmp/data.csv')
	# remove acute patietns (chronicity < 3 months)
	df = df.loc[df['Chronicity']>2].reset_index()

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

	#RAVLT learning Norms, from trial 1 to trial 5

	#trial 1
	RAVLT_T1_norm = {
	'59': {'mean': 6.8, 'sd': 1.6},
	'64': {'mean': 6.4 , 'sd': 1.9},
	'69': {'mean': 5.7, 'sd': 1.6},
	'74': {'mean': 5.5, 'sd': 1.5},
	'79': {'mean': 5.0 , 'sd': 1.5},
	'84': {'mean': 4.4 , 'sd': 1.5},
	'85': {'mean': 4.0 , 'sd': 1.8},
	}

	#trial 2
	RAVLT_T2_norm = {
	'59': {'mean': 9.5, 'sd': 2.2},
	'64': {'mean': 9.0 , 'sd': 2.3},
	'69': {'mean': 8.6, 'sd': 2.1},
	'74': {'mean': 7.8, 'sd': 1.8},
	'79': {'mean': 7.0 , 'sd': 1.9},
	'84': {'mean': 6.5 , 'sd': 1.5},
	'85': {'mean': 6.0 , 'sd': 1.8},
	}

	#trial 3
	RAVLT_T3_norm = {
	'59': {'mean': 11.4, 'sd': 2.0},
	'64': {'mean': 10.6 , 'sd': 2.3},
	'69': {'mean': 9.7, 'sd': 2.3},
	'74': {'mean': 9.1, 'sd': 2.1},
	'79': {'mean': 8.2 , 'sd': 2.2},
	'84': {'mean': 7.7 , 'sd': 2.1},
	'85': {'mean': 7.4 , 'sd': 2.2},
	}

	#trial 4
	RAVLT_T4_norm = {
	'59': {'mean': 12.4, 'sd': 1.9},
	'64': {'mean': 11.7 , 'sd': 2.7},
	'69': {'mean': 10.6, 'sd': 2.4},
	'74': {'mean': 10.2, 'sd': 2.4},
	'79': {'mean': 9.2 , 'sd': 2.2},
	'84': {'mean': 8.6 , 'sd': 2.5},
	'85': {'mean': 7.9 , 'sd': 2.4},
	}

	#trial 5
	RAVLT_T5_norm = {
	'59': {'mean': 13.1, 'sd': 1.9},
	'64': {'mean': 11.9 , 'sd': 2.0},
	'69': {'mean': 11.2, 'sd': 2.4},
	'74': {'mean': 10.5, 'sd': 2.6},
	'79': {'mean': 10.1 , 'sd': 2.2},
	'84': {'mean': 9.0 , 'sd': 2.5},
	'85': {'mean': 9.1 , 'sd': 2.3},
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

		if df.loc[i, 'Age'] <= 59:
			df.loc[i, 'RAVLT_T1_z'] = (df.loc[i, 'RAVLT_T1'] - RAVLT_T1_norm['59']['mean']) / RAVLT_T1_norm['59']['sd']
		elif 59 < df.loc[i, 'Age'] <= 64:
			df.loc[i, 'RAVLT_T1_z'] = (df.loc[i, 'RAVLT_T1'] - RAVLT_T1_norm['64']['mean']) / RAVLT_T1_norm['64']['sd']
		elif 64 < df.loc[i, 'Age'] <= 69:
			df.loc[i, 'RAVLT_T1_z'] = (df.loc[i, 'RAVLT_T1'] - RAVLT_T1_norm['69']['mean']) / RAVLT_T1_norm['69']['sd']
		elif 69 < df.loc[i, 'Age'] <= 74:
			df.loc[i, 'RAVLT_T1_z'] = (df.loc[i, 'RAVLT_T1'] - RAVLT_T1_norm['74']['mean']) / RAVLT_T1_norm['74']['sd']
		elif 74 < df.loc[i, 'Age'] <= 79:
			df.loc[i, 'RAVLT_T1_z'] = (df.loc[i, 'RAVLT_T1'] - RAVLT_T1_norm['79']['mean']) / RAVLT_T1_norm['79']['sd']
		elif 79 < df.loc[i, 'Age'] <= 84:
			df.loc[i, 'RAVLT_T1_z'] = (df.loc[i, 'RAVLT_T1'] - RAVLT_T1_norm['84']['mean']) / RAVLT_T1_norm['84']['sd']
		elif 84 < df.loc[i, 'Age']:
			df.loc[i, 'RAVLT_T1_z'] = (df.loc[i, 'RAVLT_T1'] - RAVLT_T1_norm['85']['mean']) / RAVLT_T1_norm['85']['sd']

		if df.loc[i, 'Age'] <= 59:
			df.loc[i, 'RAVLT_T2_z'] = (df.loc[i, 'RAVLT_T2'] - RAVLT_T2_norm['59']['mean']) / RAVLT_T2_norm['59']['sd']
		elif 59 < df.loc[i, 'Age'] <= 64:
			df.loc[i, 'RAVLT_T2_z'] = (df.loc[i, 'RAVLT_T2'] - RAVLT_T2_norm['64']['mean']) / RAVLT_T2_norm['64']['sd']
		elif 64 < df.loc[i, 'Age'] <= 69:
			df.loc[i, 'RAVLT_T2_z'] = (df.loc[i, 'RAVLT_T2'] - RAVLT_T2_norm['69']['mean']) / RAVLT_T2_norm['69']['sd']
		elif 69 < df.loc[i, 'Age'] <= 74:
			df.loc[i, 'RAVLT_T2_z'] = (df.loc[i, 'RAVLT_T2'] - RAVLT_T2_norm['74']['mean']) / RAVLT_T2_norm['74']['sd']
		elif 74 < df.loc[i, 'Age'] <= 79:
			df.loc[i, 'RAVLT_T2_z'] = (df.loc[i, 'RAVLT_T2'] - RAVLT_T2_norm['79']['mean']) / RAVLT_T2_norm['79']['sd']
		elif 79 < df.loc[i, 'Age'] <= 84:
			df.loc[i, 'RAVLT_T2_z'] = (df.loc[i, 'RAVLT_T2'] - RAVLT_T2_norm['84']['mean']) / RAVLT_T2_norm['84']['sd']
		elif 84 < df.loc[i, 'Age']:
			df.loc[i, 'RAVLT_T2_z'] = (df.loc[i, 'RAVLT_T2'] - RAVLT_T2_norm['85']['mean']) / RAVLT_T2_norm['85']['sd']

		if df.loc[i, 'Age'] <= 59:
			df.loc[i, 'RAVLT_T3_z'] = (df.loc[i, 'RAVLT_T3'] - RAVLT_T3_norm['59']['mean']) / RAVLT_T3_norm['59']['sd']
		elif 59 < df.loc[i, 'Age'] <= 64:
			df.loc[i, 'RAVLT_T3_z'] = (df.loc[i, 'RAVLT_T3'] - RAVLT_T3_norm['64']['mean']) / RAVLT_T3_norm['64']['sd']
		elif 64 < df.loc[i, 'Age'] <= 69:
			df.loc[i, 'RAVLT_T3_z'] = (df.loc[i, 'RAVLT_T3'] - RAVLT_T3_norm['69']['mean']) / RAVLT_T3_norm['69']['sd']
		elif 69 < df.loc[i, 'Age'] <= 74:
			df.loc[i, 'RAVLT_T3_z'] = (df.loc[i, 'RAVLT_T3'] - RAVLT_T3_norm['74']['mean']) / RAVLT_T3_norm['74']['sd']
		elif 74 < df.loc[i, 'Age'] <= 79:
			df.loc[i, 'RAVLT_T3_z'] = (df.loc[i, 'RAVLT_T3'] - RAVLT_T3_norm['79']['mean']) / RAVLT_T3_norm['79']['sd']
		elif 79 < df.loc[i, 'Age'] <= 84:
			df.loc[i, 'RAVLT_T3_z'] = (df.loc[i, 'RAVLT_T3'] - RAVLT_T3_norm['84']['mean']) / RAVLT_T3_norm['84']['sd']
		elif 84 < df.loc[i, 'Age']:
			df.loc[i, 'RAVLT_T3_z'] = (df.loc[i, 'RAVLT_T3'] - RAVLT_T3_norm['85']['mean']) / RAVLT_T3_norm['85']['sd']

		if df.loc[i, 'Age'] <= 59:
			df.loc[i, 'RAVLT_T4_z'] = (df.loc[i, 'RAVLT_T4'] - RAVLT_T4_norm['59']['mean']) / RAVLT_T4_norm['59']['sd']
		elif 59 < df.loc[i, 'Age'] <= 64:
			df.loc[i, 'RAVLT_T4_z'] = (df.loc[i, 'RAVLT_T4'] - RAVLT_T4_norm['64']['mean']) / RAVLT_T4_norm['64']['sd']
		elif 64 < df.loc[i, 'Age'] <= 69:
			df.loc[i, 'RAVLT_T4_z'] = (df.loc[i, 'RAVLT_T4'] - RAVLT_T4_norm['69']['mean']) / RAVLT_T4_norm['69']['sd']
		elif 69 < df.loc[i, 'Age'] <= 74:
			df.loc[i, 'RAVLT_T4_z'] = (df.loc[i, 'RAVLT_T4'] - RAVLT_T4_norm['74']['mean']) / RAVLT_T4_norm['74']['sd']
		elif 74 < df.loc[i, 'Age'] <= 79:
			df.loc[i, 'RAVLT_T4_z'] = (df.loc[i, 'RAVLT_T4'] - RAVLT_T4_norm['79']['mean']) / RAVLT_T4_norm['79']['sd']
		elif 79 < df.loc[i, 'Age'] <= 84:
			df.loc[i, 'RAVLT_T4_z'] = (df.loc[i, 'RAVLT_T4'] - RAVLT_T4_norm['84']['mean']) / RAVLT_T4_norm['84']['sd']
		elif 84 < df.loc[i, 'Age']:
			df.loc[i, 'RAVLT_T4_z'] = (df.loc[i, 'RAVLT_T4'] - RAVLT_T4_norm['85']['mean']) / RAVLT_T4_norm['85']['sd']

		if df.loc[i, 'Age'] <= 59:
			df.loc[i, 'RAVLT_T5_z'] = (df.loc[i, 'RAVLT_T5'] - RAVLT_T5_norm['59']['mean']) / RAVLT_T5_norm['59']['sd']
		elif 59 < df.loc[i, 'Age'] <= 64:
			df.loc[i, 'RAVLT_T5_z'] = (df.loc[i, 'RAVLT_T5'] - RAVLT_T5_norm['64']['mean']) / RAVLT_T5_norm['64']['sd']
		elif 64 < df.loc[i, 'Age'] <= 69:
			df.loc[i, 'RAVLT_T5_z'] = (df.loc[i, 'RAVLT_T5'] - RAVLT_T5_norm['69']['mean']) / RAVLT_T5_norm['69']['sd']
		elif 69 < df.loc[i, 'Age'] <= 74:
			df.loc[i, 'RAVLT_T5_z'] = (df.loc[i, 'RAVLT_T5'] - RAVLT_T5_norm['74']['mean']) / RAVLT_T5_norm['74']['sd']
		elif 74 < df.loc[i, 'Age'] <= 79:
			df.loc[i, 'RAVLT_T5_z'] = (df.loc[i, 'RAVLT_T5'] - RAVLT_T5_norm['79']['mean']) / RAVLT_T5_norm['79']['sd']
		elif 79 < df.loc[i, 'Age'] <= 84:
			df.loc[i, 'RAVLT_T5_z'] = (df.loc[i, 'RAVLT_T5'] - RAVLT_T5_norm['84']['mean']) / RAVLT_T5_norm['84']['sd']
		elif 84 < df.loc[i, 'Age']:
			df.loc[i, 'RAVLT_T5_z'] = (df.loc[i, 'RAVLT_T5'] - RAVLT_T5_norm['85']['mean']) / RAVLT_T5_norm['85']['sd']



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


def Cal_lesion_size():
	''' Calculate lesion size '''

	df = pd.read_csv('~/RDSS/tmp/data_z.csv')

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


def determine_comparison_patients():
	'''Check comparison pateitns lesion overlap with LESYMAP results, select appropriate controls '''

	########################################################################
	df = pd.read_csv('~/RDSS/tmp/data_z.csv')

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

	# Rey trial 1 and other scores have no significant LESYMAP clusters
	REY2_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/REY_2/stat_img.nii.gz')
	REY2_LESYMAP_map = REY2_LESYMAP.get_data()
	REY3_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/REY_3/stat_img.nii.gz')
	REY3_LESYMAP_map = REY3_LESYMAP.get_data()
	REY4_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/REY_4/stat_img.nii.gz')
	REY4_LESYMAP_map = REY4_LESYMAP.get_data()
	REY5_LESYMAP = nib.load('/home/kahwang/LESYMAP_for_Kai/REY_5/stat_img.nii.gz')
	REY5_LESYMAP_map = REY5_LESYMAP.get_data()

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

			if np.sum(res_m*REY2_LESYMAP_map)==0:
				df.loc[i, 'RAVLT_T2_Comparison'] = True
			else:
				df.loc[i, 'RAVLT_T2_Comparison'] = False

			if np.sum(res_m*REY3_LESYMAP_map)==0:
				df.loc[i, 'RAVLT_T3_Comparison'] = True
			else:
				df.loc[i, 'RAVLT_T3_Comparison'] = False

			if np.sum(res_m*REY4_LESYMAP_map)==0:
				df.loc[i, 'RAVLT_T4_Comparison'] = True
			else:
				df.loc[i, 'RAVLT_T4_Comparison'] = False

			if np.sum(res_m*REY5_LESYMAP_map)==0:
				df.loc[i, 'RAVLT_T5_Comparison'] = True
			else:
				df.loc[i, 'RAVLT_T5_Comparison'] = False

			df.loc[i, 'TMTA_Comparison'] = True
			df.loc[i, 'RAVLT_T1_Comparison'] = True
			df.loc[i, 'RAVLT_Delayed_Recall_Comparison'] = True
			df.loc[i, 'RAVLT_Recognition_Comparison'] = True

		except:
			continue

	df.to_csv('~/RDSS/tmp/data_z.csv')


def neuropsych_zscore(zthreshold):
	'''determine if a task is imparied with zscore'''

	#######################################	#################################
	df = pd.read_csv('~/RDSS/tmp/data_z.csv')

	df['TMTA_z_Impaired'] = df['TMTA_z'] >-1*zthreshold
	df['TMTB_z_Impaired'] = df['TMTB_z'] >-1*zthreshold
	df['BNT_z_Impaired'] = df['BNT_z'] <zthreshold
	df['COWA_z_Impaired'] = df['COWA_z'] <zthreshold
	df['RAVLT_Delayed_Recall_z_Impaired'] = df['RAVLT_Delayed_Recall_z'] <zthreshold
	df['RAVLT_Recognition_z_Impaired'] = df['RAVLT_Recognition_z'] <zthreshold
	df['Complex_Figure_Copy_z_Impaired'] = df['Complex_Figure_Copy_z'] <zthreshold
	df['Complex_Figure_Recall_z_Impaired'] = df['Complex_Figure_Recall_z'] <zthreshold
	df['RAVLT_T1_z_Impaired'] = df['RAVLT_T1_z'] <zthreshold
	df['RAVLT_T2_z_Impaired'] = df['RAVLT_T2_z'] <zthreshold
	df['RAVLT_T3_z_Impaired'] = df['RAVLT_T3_z'] <zthreshold
	df['RAVLT_T4_z_Impaired'] = df['RAVLT_T4_z'] <zthreshold
	df['RAVLT_T5_z_Impaired'] = df['RAVLT_T5_z'] <zthreshold

	df['MM_impaired'] = sum([df['TMTB_z_Impaired'],
	np.any([df['BNT_z_Impaired'] , df['COWA_z_Impaired']], axis=0),
	np.any([df['RAVLT_Delayed_Recall_z_Impaired'] , df['RAVLT_Recognition_z_Impaired'] , df['Complex_Figure_Recall_z_Impaired']], axis=0),
	np.any([df['Complex_Figure_Copy_z_Impaired'], df['TMTA_z_Impaired']], axis=0),
	np.any([df['RAVLT_T1_z_Impaired'] + df['RAVLT_T1_z_Impaired'] + df['RAVLT_T1_z_Impaired'] + df['RAVLT_T1_z_Impaired'] + df['RAVLT_T1_z_Impaired']], axis=0)])
	df.to_csv('~/RDSS/tmp/data_z.csv')


def plot_neuropsy_indiv_comparisons():
	'''plot neuropsych scores and compare between groups'''

	list_of_neuropsych = ['TMT Part A', 'TMT Part B', 'Boston Naming', 'COWA', 'RAVLT Delayed Recall', 'RAVLT Recognition', 'RAVLT Trial 1', ' RAVLT Trial 2',
	'RAVLT Trial 3', 'RAVLT Trial 4', 'RAVLT Trial 5', 'Complex Figure Copy', 'Complex Figure Recall']

	list_of_neuropsych_z = ['TMTA_z', 'TMTB_z', 'BNT_z', 'COWA_z', 'RAVLT_Delayed_Recall_z', 'RAVLT_Recognition_z', 'RAVLT_T1_z',
	'RAVLT_T2_z', 'RAVLT_T3_z', 'RAVLT_T4_z', 'RAVLT_T5_z', 'Complex_Figure_Copy_z', 'Complex_Figure_Recall_z']

	list_of_neuropsych_comp = ['TMTA_Comparison', 'TMTB_Comparison', 'BNT_Comparison', 'COWA_Comparison', 'RAVLT_Delayed_Recall_Comparison', 'RAVLT_Recognition_Comparison', 'RAVLT_T1_Comparison',
	'RAVLT_T2_Comparison', 'RAVLT_T3_Comparison', 'RAVLT_T4_Comparison', 'RAVLT_T5_Comparison', 'Complex_Figure_Copy_Comparison', 'Complex_Figure_Recall_Comparison']

	#Need to only plot included comparison patients
	for i, test in enumerate(list_of_neuropsych_z):
		plt.figure(figsize=[2.4,3])
		sns.set_context('paper', font_scale=1)
		sns.set_style('white')
		sns.set_palette("Set1")

		tdf = df.loc[df[list_of_neuropsych_comp[i]]==True]
		fig1=sns.pointplot(x="Site", y=test, join=False, hue='Site', dodge=False,
					  data=tdf)
		fig1=sns.stripplot(x="Site", y=test,
					  data=tdf, alpha = .4)

		fig1.set_xticklabels(['Thalamic \nPatients', 'Comparison \nPatients'])
		#fig1.set_ylim([-3, 7])
		#fig1.set_aspect(.15)
		fig1.legend_.remove()
		plt.xlabel('')
		plt.ylabel(list_of_neuropsych[i])
		plt.tight_layout()
		fn = '/home/kahwang/RDSS/tmp/fig_%s.pdf' %test
		plt.savefig(fn)
		plt.close()


def plot_neuropsy_comparisons():
	'''plot neuropsych scores and compare between groups'''

	#need to melt df
	tdf = pd.melt(df, id_vars = ['Sub', 'Site'],
		value_vars = ['TMTA_z', 'Complex_Figure_Copy_z', 'TMTB_z', 'BNT_z', 'COWA_z', 'RAVLT_T1_z',
		'RAVLT_T2_z', 'RAVLT_T3_z', 'RAVLT_T4_z', 'RAVLT_T5_z', 'RAVLT_Delayed_Recall_z', 'RAVLT_Recognition_z', 'Complex_Figure_Recall_z'], value_name = 'Z Score', var_name ='Task' )


	plt.figure(figsize=[6,4])
	sns.set_context('paper', font_scale=1)
	sns.set_style('white')
	sns.set_palette("Set1")

	fig1=sns.pointplot(x="Task", y="Z Score", hue="Site",
				  data=tdf, dodge=.42, join=False)

	fig1=sns.stripplot(x="Task", y="Z Score", hue="Site",
				  data=tdf, dodge=True, alpha=.25)
	fig1.legend_.remove()
	fig1.set_ylim([-6, 6])
	fig1.set_xticklabels(['TMT \nPart A', 'Complex Figure \nConstruction', 'TMT \nPart B', 'Boston \nNaming', 'COWA', 'RAVLT Trial 1', ' RAVLT Trial 2',
	'RAVLT Trial 3', 'RAVLT Trial 4', 'RAVLT Trial 5', 'RAVLT \nRecall', 'RAVLT \nRecognition',  'Comeplex Figure \nRecall'], rotation=90)
	plt.xlabel('')
	#plt.show()
	plt.tight_layout()
	fn = '/home/kahwang/RDSS/tmp/neuropych.pdf'
	plt.savefig(fn)


def cal_neuropsych_z_anchor_to_controls():
	''' Calculate neuropsych z scores using mean and SD from the comparison group'''
	df = pd.read_csv('~/RDSS/tmp/data_z.csv')

	list_of_neuropsych_z = ['TMTB', 'BNT', 'COWA', 'RAVLT_Delayed_Recall', 'RAVLT_Recognition', 'Complex_Figure_Copy',
	 'Complex_Figure_Recall', 'RAVLT_T1', 'RAVLT_T2', 'RAVLT_T2', 'RAVLT_T3', 'RAVLT_T4', 'RAVLT_T5']

	for neuropsych in list_of_neuropsych_z:
		tfn = neuropsych+'_z'
		tcomp = neuropsych+'_Comparison'
		tfnvar = neuropsych+'_z_c'
		for i in df.index:
			df.loc[i, tfnvar] = (df.loc[i, tfn] - np.nanmean(df.loc[(df['Site']=='ctx')& (df[tcomp]==True)][tfn])) / np.nanstd(df.loc[(df['Site']=='ctx')& (df[tcomp]==True)][tfn])
	df.to_csv('~/RDSS/tmp/data_z.csv')


def plot_neuropsych_table():
	''' table of task impairment'''
	ddf = df[['SubID','TMTA_z', 'Complex_Figure_Copy_z', 'TMTB_z', 'BNT_z', 'COWA_z',
		'RAVLT_T1_z', 'RAVLT_T2_z', 'RAVLT_T3_z', 'RAVLT_T4_z', 'RAVLT_T5_z',
		'RAVLT_Delayed_Recall_z', 'RAVLT_Recognition_z', 'Complex_Figure_Recall_z','MM_impaired']]
	tddf = ddf.loc[df['Site']=='Th']
	#invert tmtbz
	tddf['TMTB_z'] = tddf['TMTB_z']*-1

	tddf = tddf.set_index('SubID')
	figt = sns.heatmap(tddf.sort_values('MM_impaired'), vmin = -5, vmax=5, center=0, cmap="coolwarm")
	figt.set_xticklabels(['TMT Part A', 'Complex Figure Construction',  'TMT Part B', 'Boston Naming', 'COWA',
	'RAVLT Trial 1', 'RAVLT Trial 2', 'RAVLT Trial 3', 'RAVLT Trial 4', 'RAVLT Trial 5',
	'RAVLT Recall', 'RAVLT Recognition', 'Comeplex Figure Recall', '# Tasks Impaired'], rotation=90)
	plt.xlabel('')
	plt.ylabel('Patient')
	plt.tight_layout()
	fn = '/home/kahwang/RDSS/tmp/tasktable.pdf'
	plt.savefig(fn)


def draw_lesion_overlap(group):
	''' draw lesion overlap, group ='Th' or 'ctx' '''

	m=0
	for s in df.loc[df['Site']==group]['Sub']:
		# load mask and get size

		# annoying zeros
		if s == '902':
			s = '0902'
		if s == '802':
			s = '0802'
		try:
			fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
			m = m + nib.load(fn).get_fdata()
		except:
			continue

	h = nib.load('/home/kahwang/0.5mm/0902.nii.gz')
	lesion_overlap_nii = nilearn.image.new_img_like(h, m)
	#lesion_overlap_nii.to_filename('lesion_overlap.nii')

	return lesion_overlap_nii


if __name__ == "__main__":

	########################################################################
	# Figure 1. Compare test scores between patient groups
	########################################################################
	### Prep dataframe through steps:

	#load_and_normalize_neuropsych_data()
	#Cal_lesion_size()
	#determine_comparison_patients()
	#neuropsych_zscore()

	###################
	# compare test scores
	df = pd.read_csv('~/RDSS/tmp/data_z.csv')

	### t tests

	# visual-motor, Construction
	print('TMTA')
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx') & (df['TMTB_Comparison']==True)]['TMTA_z'].values, df.loc[df['Site']=='Th']['TMTA_z'].values))
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx') & (df['Complex_Figure_Copy_Comparison']==True)]['Complex_Figure_Copy_z'].values, df.loc[df['Site']=='Th']['Complex_Figure_Copy_z'].values))

	# executive function
	print('TMTB')
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx') & (df['TMTB_Comparison']==True)]['TMTB_z'].values, df.loc[df['Site']=='Th']['TMTB_z'].values))

	# Language
	print('BNT')
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx') & (df['BNT_Comparison']==True)]['BNT_z'].values, df.loc[df['Site']=='Th']['BNT_z'].values))
	print('COWA')
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx') & (df['COWA_Comparison']==True)]['COWA_z'].values, df.loc[df['Site']=='Th']['COWA_z'].values))

	# learning, long-term memory recall
	print('RAVLT recall')
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['RAVLT_Delayed_Recall_z'].values, df.loc[df['Site']=='Th']['RAVLT_Delayed_Recall_z'].values))
	print('RAVLT, recog')
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['RAVLT_Recognition_z'].values, df.loc[df['Site']=='Th']['RAVLT_Recognition_z'].values))

	print('RAVLT, learning trials')
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')]['RAVLT_T1_z'].values, df.loc[df['Site']=='Th']['RAVLT_T1_z'].values))
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')& (df['RAVLT_T2_Comparison']==True)]['RAVLT_T2_z'].values, df.loc[df['Site']=='Th']['RAVLT_T2_z'].values))
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')& (df['RAVLT_T3_Comparison']==True)]['RAVLT_T3_z'].values, df.loc[df['Site']=='Th']['RAVLT_T3_z'].values))
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')& (df['RAVLT_T4_Comparison']==True)]['RAVLT_T4_z'].values, df.loc[df['Site']=='Th']['RAVLT_T4_z'].values))
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx')& (df['RAVLT_T5_Comparison']==True)]['RAVLT_T5_z'].values, df.loc[df['Site']=='Th']['RAVLT_T5_z'].values))

	print('complex figure, copy and recall')
	print(scipy.stats.mannwhitneyu(df.loc[(df['Site']=='ctx') & (df['Complex_Figure_Recall_Comparison']==True)]['Complex_Figure_Recall_z'].values, df.loc[df['Site']=='Th']['Complex_Figure_Recall_z'].values))

	#plot_neuropsy_indiv_comparisons()
	#plot_neuropsy_comparisons()

	###################
	# plot lesion overlap
	lesion_overlap_nii = draw_lesion_overlap('Th')
	lesion_overlap_nii.to_filename('Th_lesion_overlap.nii')
	#plotting.plot_stat_map(lesion_overlap_nii, bg_img = mni_template, display_mode='z', cut_coords=5, colorbar = False, black_bg=False, cmap='gist_ncar')
	#plotting.show()

	lesion_overlap_nii = draw_lesion_overlap('ctx')
	lesion_overlap_nii.to_filename('Ctx_lesion_overlap.nii')
	#plotting.plot_stat_map(lesion_overlap_nii, bg_img = mni_template, display_mode='z', cut_coords=5, colorbar = False, black_bg=False, cmap='gist_ncar')
	#plotting.show()

	################################
	# Figure 2.  Plot table of z scores to show mutlimodal impairment
	################################

	#plot_neuropsych_table()


	##### Now draw lesions overlap for patients with and without multimodal impairment
	##### plot lesion masks for these subjects

	# plot lesion overlap of patients that show MM lesion or SM lesions
	def plt_MM_SM_lesion_mask():
		m=0
		for s in ['2105', '2552', '2092', 'ca085', 'ca093', 'ca104', 'ca105', '1692', '1830', 3049]:

			fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
			m = m + nib.load(fn).get_fdata()

		h = nib.load('/home/kahwang/0.5mm/0902.nii.gz')
		mmlesion_overlap_nii = nilearn.image.new_img_like(h, m)
		mmlesion_overlap_nii.to_filename('images/mmlesion_overlap.nii.gz')

		m=0
		for s in ['1809', '1105', '2781', '0902', '3184', 'ca018', 'ca041', '4036', '4032', '4041']:

			fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
			m = m + nib.load(fn).get_fdata()

		h = nib.load('/home/kahwang/0.5mm/0902.nii.gz')
		smlesion_overlap_nii = nilearn.image.new_img_like(h, m)
		smlesion_overlap_nii.to_filename('images/smlesion_overlap.nii.gz')


	def map_lesion_unique_masks(show_plot = False):
		''' map each neuropsych's unique lesion mask'''

		thalamus_mask_data = nib.load('/home/kahwang/0.5mm/tha_0.5_mask.nii.gz').get_fdata()
		thalamus_mask_data = thalamus_mask_data>0
		list_of_neuropsych_z = ['TMTA', 'TMTB', 'BNT', 'COWA', 'RAVLT_Delayed_Recall', 'RAVLT_Recognition', 'Complex_Figure_Copy',
		 'Complex_Figure_Recall', 'RAVLT_T1', 'RAVLT_T2', 'RAVLT_T2', 'RAVLT_T3', 'RAVLT_T4', 'RAVLT_T5']
		list_of_neuropsych_var = ['TMT part A', 'TMT part B', 'Boston Naming', 'COWA', 'RAVLT Recall', 'RAVLT Recognition', 'Complex Figure Construction',
 		 'Complex Figure Recall', 'RAVLT Trial 1', 'RAVLT Trial 2', 'RAVLT Trial 2', 'RAVLT Trial 3', 'RAVLT Trial 4', 'RAVLT Trial 5']
		h = nib.load('/home/kahwang/0.5mm/0902.nii.gz')

		mask_niis = {}
		for j, neuropsych in enumerate(list_of_neuropsych_z):
			strc = neuropsych + '_z_Impaired'
			tdf = df.loc[df['Site']=='Th'].loc[df['MNI_Mask']=='T'].loc[df[strc]==True]
			num_patient = len(tdf) # number of patients
			if num_patient ==0:
				continue # break loop if no impairment
			tdf.loc[tdf['Sub']=='902','Sub'] = '0902'

			#lesion overlap of patients with impariment
			true_m=0
			for i in tdf.index:
				s = tdf.loc[i, 'Sub']
				fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
				true_m = true_m + nib.load(fn).get_fdata()
			#true_m = 1.0 * (true_m>( 0* num_patient))  # find "union" voxels

			#true_m = (true_m * 1.0) / num_patient

			#tdf = df.loc[df['Site']=='Th'].loc[df['MNI_Mask']=='T'].loc[df[strc]==False]
			#num_patient = len(tdf)
			#tdf.loc[tdf['Sub']=='902','Sub'] = '0902'

			#lesion overlap of patients with no impariment
			# false_m=0
			# for i in tdf.index:
			# 	s = tdf.loc[i, 'Sub']
			# 	fn = '/home/kahwang/0.5mm/%s.nii.gz' %s
			# 	false_m = false_m + nib.load(fn).get_fdata()
			# false_m = 1.0 * (false_m>(0 * num_patient))
			#
			# # diff of these two lesion masks
			# diff_m = 1.0*(true_m>0) - 1.0*(false_m>0)
			true_m = true_m * thalamus_mask_data
			impairment_overlap_nii = nilearn.image.new_img_like(h, true_m)
			#Noimpairment_overlap_nii = nilearn.image.new_img_like(h, false_m)
			#Diff_nii = nilearn.image.new_img_like(h, diff_m)

			fn = 'images/' + neuropsych + '_lesionmask_pcount.nii.gz'
			impairment_overlap_nii.to_filename(fn)

			if show_plot:
				plotting.plot_stat_map(impairment_overlap_nii, title = list_of_neuropsych_var[j], cmap='black_purple', vmax = 3, cut_coords =[7,11], display_mode = 'z')
				plotting.show()

			mask_niis[list_of_neuropsych_var[j]] = impairment_overlap_nii

			#cosolidate masks
			# or use terminal, do 3dcalc -a BNT_lesionmask_impaired.percentage.overlap.nii.gz -b COWA_lesionmask_impaired.percentage.overlap.nii.gz -expr '(a+b)/2' -prefix language_impairment_mask.nii.gz


		VM_mask = np.max([mask_niis['TMT part A'].get_fdata(), mask_niis['Complex Figure Construction'].get_fdata()], axis=0)
		RAVLT_mask = np.max([mask_niis['RAVLT Trial 1'].get_fdata(),
			mask_niis['RAVLT Trial 2'].get_fdata(),
			mask_niis['RAVLT Trial 3'].get_fdata(),
			mask_niis['RAVLT Trial 4'].get_fdata(),
			mask_niis['RAVLT Trial 5'].get_fdata()], axis=0)

		Verbal_mask = np.max([mask_niis['Boston Naming'].get_fdata(), mask_niis['COWA'].get_fdata()], axis=0)
		Memory_mask = np.max([mask_niis['RAVLT Recall'].get_fdata(), mask_niis['RAVLT Recognition'].get_fdata()], axis=0)
		TMTB_mask = mask_niis['TMT part B'].get_fdata()

		RAVLT_mask = nilearn.image.new_img_like(h, RAVLT_mask)
		RAVLT_mask.to_filename('images/RAVLT_overlap.nii.gz')
		Verbal_mask = nilearn.image.new_img_like(h, Verbal_mask)
		Verbal_mask.to_filename('images/Verbal_overlap.nii.gz')
		VM_mask = nilearn.image.new_img_like(h, VM_mask)
		VM_mask.to_filename('images/VM_overlap.nii.gz')
		Memory_mask = nilearn.image.new_img_like(h, Memory_mask)
		Memory_mask.to_filename('images/Memory_overlap.nii.gz')
		TMTB_mask = nilearn.image.new_img_like(h, TMTB_mask)
		TMTB_mask.to_filename('images/TMTB_overlap.nii.gz')

		# count number of task lesion mask overlap
		Num_task_mask = 1.0*(RAVLT_mask.get_fdata()>0) + 1.0*(Verbal_mask.get_fdata()>0) + 1.0*(VM_mask.get_fdata()>0) + 1.0*(TMTB_mask.get_fdata()>0) + 1.0*(Memory_mask.get_fdata()>0)
		Num_task_mask = nilearn.image.new_img_like(h, Num_task_mask)
		Num_task_mask.to_filename('images/Num_of_task_impaired_overlap.nii.gz')
		#map_lesion_unique_masks(show_plot = False)



	################################
	# Figure SX.  Correlation among scores
	################################
	def plot_corr_table():
		corrdf = pd.read_csv('~/RDSS/tmp/crosscorr.csv')
		corrdf = corrdf.set_index('Tasks')
		corrplot = sns.heatmap(corrdf, vmin = -1, vmax=1, center=0, cmap="coolwarm")
		corrplot.set_xticklabels(['Lesion Size', 'TMT Part A','TMT Part B', 'Boston Naming', 'COWA',
		'RAVLT Recall', 'RAVLT Recognition',  'RAVLT Trial 1', 'RAVLT Trial 2', 'RAVLT Trial 3', 'RAVLT Trial 4', 'RAVLT Trial 5',
		'Complex Figure Copy', 'Comeplex Figure Recall'], rotation=90)
		corrplot.set_yticklabels(['Lesion Size', 'TMT Part A','TMT Part B', 'Boston Naming', 'COWA',
		'RAVLT Recall', 'RAVLT Recognition',  'RAVLT Trial 1', 'RAVLT Trial 2', 'RAVLT Trial 3', 'RAVLT Trial 4', 'RAVLT Trial 5',
		'Complex Figure Copy', 'Comeplex Figure Recall'])
		plt.xlabel('')
		plt.ylabel('')
		plt.tight_layout()
		#plt.show()
		#plotting.show()
		fn = '/home/kahwang/RDSS/tmp/corrtable.pdf'
		plt.savefig(fn)


	################################
	# Figure 3  Compare lesion site to PC values
	################################





#end of line
