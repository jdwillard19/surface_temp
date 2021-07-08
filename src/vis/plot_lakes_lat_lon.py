import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

metadata2 = pd.read_csv("../../metadata/lake_metadata_full_conus_185k.csv")
# pdb.set_trace()
# metadata_old = 
# df['diff'] = df['rmse_ealstm'] - df['rmse_xgboost']
df = pd.DataFrame()
df['x'] = metadata2['lon']
# df['x'] = metadata['elevation']
df['y'] = metadata2['lat']
df['site_id'] = metadata2['site_id']
# df['y'] = np.log(metadata['area_m2'])
# df['log area'] = np.log(metadata['area_m2'])

cts = np.load("../../data/raw/obs/sites_with_cold_temps_ct.npy",allow_pickle=True)
sites = np.load("../../data/raw/obs/sites_with_cold_temps.npy",allow_pickle=True)
sites = sites[cts > 3]

# df['isWorse'] = df['rmse_ealstm'] > df['rmse_xgboost']
# pdb.set_trace()
cold_df = df[np.isin(df['site_id'].values,sites)]
fig, ax = plt.subplots()
# ax.scatter(z, y)


# plt.scatter(df[df['diff'] > .5]['x'], df[df['diff'] > .5]['y'],color='red',label='EA-LSTM worse by .5',s=2)
plt.scatter(df['x'].values, df['y'].values,color='black',s=1,alpha=.05,label='rest of lakes')
plt.scatter(cold_df['x'].values, cold_df['y'].values,color='red',s=3,alpha=.5, label='contain < 8 deg C obs June-Sept')
print(cold_df['site_id'])
for i, txt in enumerate(cold_df['site_id'].values):
    ax.annotate(txt, (cold_df['x'].values[i], cold_df['y'].values[i]),color='red',fontsize='xx-small')
# plt.scatter(df[df['diff'] < -.5]['x'], df[df['diff'] < -.5]['y'],color='blue',label='EA-LSTM better by .5',s=2)
# plt.scatter(df['x'], df['y'],color='blue',label='EA-LSTM better by .5',s=2)
plt.ylabel('longitude')
plt.xlabel('latitude')
plt.legend()
plt.show()