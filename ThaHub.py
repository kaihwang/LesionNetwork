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


# normalize neuropsych data to popluation norm

# compare neuropsych data
#BNT norms:
# 18-39 55.8 (3.8)
# 40-49 56.8 (3)
# 50-59 55.2 (4)
# 60-69 53.3 (4.6)
# 70-79 48.9 (6.3

#COWA norms:
#M, 12  36.9 (9,8)
#   13-15   40.5 (9.4)
#   16  41 (9.8)
#F, 12  35.9 (9.6)
#   13-35   39.4 (10.1)
#   16  46.5(11.2)

# RVLT Delay recall
#   55-59   10.4 (3.1)
#   60-64   9.9 (3.1)
#   65-69   8.3 (3.5)
#   70-74   7.4 (3.1)
#   75-79   6.9 (2.9)
#   80-84   5.5 (3.3)
#   85-     5.4 (2.7)

# RVLT  Recognition
#   55-59   14 (1.3)
#   60-64   13.9 (1.5)
#   65-69   13.3 (2)
#   70-74   12.7 (2.1)
#   75-79   12.5 (2.4)
#   80-84   12.3 (2.4)
#   85-     12.3 (2.3)

# rey o complex figure construction
#   40  32.83 (3.1)
#   50  31.79 (4.55)
#   60  31.94 (3.37)
#   70  31.76 (3.64)
#   80  30.14 (4.52)


# rey o complex figure delayed recall
#   40  19.28 (7.29)
#   50  17.13 (7.14)
#   60  16.55 (6.08)
#   70  15.18 (5.57)
#   80  13.17 (5.32)



	for i, s in enumerate(df['Sub']):

		#convert score
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
