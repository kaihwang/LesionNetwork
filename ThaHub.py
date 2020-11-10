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

# calculate lesion size
df = pd.read_csv('~/RDSS/tmp/data.csv')



# compare neuropsych data
