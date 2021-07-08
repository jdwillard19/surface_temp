import pandas as pd
import numpy as np
import pdb
import sys
import plotly.express as px
from scipy import stats
import matplotlib.pyplot as plt

import os
from sklearn.cluster import KMeans

metadata = pd.read_csv("../../metadata/surface_lake_metadata_021521_wCluster.csv")
# metadata = pd.read_csv("../../metadata/surface_lake_metadata_file_temp.csv
cluster_label_vals = [int(i) for i in range(1,16)]
err_data = pd.read_feather("../../results/err_per_site_031821.feather")
obs_data = pd.read_feather("../../results/all_outputs_and_obs_031821.feather")

pdb.set_trace()
df = pd.DataFrame()
df['cluster'] = cluster_label_vals
df['Median RMSE'] = [np.median(err_data[metadata['cluster']==i]['rmse_ealstm']) for i in cluster_label_vals]




# fig = px.bar(df, x='cluster', y='Median RMSE')
# fig.update_layout(uniformtext_minsize=20, uniformtext_mode='hide')
# fig.update_traces(textposition='inside', textfont_size=20)
# fig.show()

plt.bar(df['cluster'], df['Median RMSE'], color='blue')
plt.xlabel("Cluster")
plt.ylabel("Median RMSE")
plt.title("RMSE per Cluster")
plt.show()
# print([stats.percentileofscore(metadata['area_m2'].values, np.median(metadata[metadata['cluster']==i]['area_m2'])) for i in cluster_label_vals])
# print([np.median(metadata[metadata['cluster']==i]['area_m2']) for i in cluster_label_vals])
# sys.exit()
# for i in cluster_label_vals:
# 	metadata['isCluster'+str(i)] = (metadata['cluster'] == i)
# 	fig = px.scatter(metadata, x='lon', y='lat',color='isCluster'+str(i),color_continuous_scale=px.colors.smoker[::-1])
# 	# fig = px.scatter(metadata[metadata['cluster']==i], x='lon', y='lat',color='cluster')
# 	# fig = px.scatter(metadata[metadata['cluster'] != i], x='lon', y='lat')
# 	fig.update_traces(marker=dict(size=5),
#                   selector=dict(mode='markers'))
# 	fig.write_html("./cluster/clustered_lakes-conus_cluster_"+str(i)+".html")


