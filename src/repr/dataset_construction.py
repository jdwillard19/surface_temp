import pandas as pd
import numpy as np
import sys
import os




#load lakes
site_ids = np.load("../../metadata/lakeset.npy")

#load obs
obs = pd.read


#create train/test split
def createBinID(date):
site_df[(site_df['Date'].str.contains('-12-')) | (site_df['Date'].str.contains('-01-')) | (site_df['Date'].str.contains('-02-'))]