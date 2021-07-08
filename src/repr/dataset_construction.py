import pandas as pd
import numpy as np
import sys
import os




#load lakes
site_ids = np.load("../../metadata/lakeset.npy",allow_pickle=True)

#load obs
obs = pd.read_csv("lake_surface_temp_obs.csv")

#trim obs to sites
pdb.set_trace()
# 
#create train/test split
# def createBinID(date):

# site_df[(site_df['Date'].str.contains('-12-')) | (site_df['Date'].str.contains('-01-')) | (site_df['Date'].str.contains('-02-'))]
 #year since 1980 * 4 + season (winter1,spring2,summ3,fall4)


 # for obs in obs