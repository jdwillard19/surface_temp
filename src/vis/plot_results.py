import numpy as np
import pdb
import sys
import os
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
# df = pd.read_feather("../../results/err_per_site.feather")

# metadata = pd.read_csv("../metadata/surface_lake_metadata_021521_wCluster.csv")
metadata = pd.read_csv("../../metadata/surface_lake_metadata_021521.csv")
metadata2 = pd.read_csv("../../metadata/lake_metadata_full_conus_185k.csv")
# pdb.set_trace()
# metadata_old = 
# df['diff'] = df['rmse_ealstm'] - df['rmse_xgboost']
df = pd.DataFrame()
df['x'] = metadata2['lon']
# df['x'] = metadata['elevation']
df['y'] = metadata2['lat']
# df['y'] = np.log(metadata['area_m2'])
# df['log area'] = np.log(metadata['area_m2'])


# df['isWorse'] = df['rmse_ealstm'] > df['rmse_xgboost']

# plt.scatter(df[df['diff'] > .5]['x'], df[df['diff'] > .5]['y'],color='red',label='EA-LSTM worse by .5',s=2)
plt.scatter(df['x'].values, df['y'].values,color='black',s=1,alpha=.05)
# plt.scatter(df[df['diff'] < -.5]['x'], df[df['diff'] < -.5]['y'],color='blue',label='EA-LSTM better by .5',s=2)
# plt.scatter(df['x'], df['y'],color='blue',label='EA-LSTM better by .5',s=2)
plt.ylabel('longitude')
plt.xlabel('latitude')
plt.legend()
plt.show()

# fig = px.scatter(df, x='rmse_ealstm', y='rmse_xgboost',color='log area')
# fig.show()
