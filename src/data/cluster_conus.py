import pandas as pd
import numpy as np
import pdb
import sys
import os
import plotly.express as px
from sklearn.cluster import KMeans

metadata = pd.read_csv("../../metadata/surface_lake_metadata_conus.csv")
obs = pd.read_feather("../../data/raw/obs/temp_wqp_munged.feather")

site_ids = np.unique(obs['site_id'].values)
metadata = metadata[np.isin(metadata['site_id'],site_ids)]
metadata['log_area'] = np.log(metadata['area_m2'].values)

normalize = True
if normalize:
	metadata['lat'] = (metadata['lat'].values - metadata['lat'].values.mean()) / metadata['lat'].values.std()
	metadata['lon'] = (metadata['lon'].values - metadata['lon'].values.mean()) / metadata['lon'].values.std()
	metadata['log_area'] = (metadata['log_area'].values - metadata['log_area'].values.mean()) / metadata['log_area'].values.std()
print("starting clustering...")
n_clusters = 16
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(metadata[['lat','lon','log_area']].values)
print("clustering done!")
metadata['cluster'] = kmeans.labels_
# fig = px.scatter_3d(metadata, x='lon', y='lat', z='log_area',color='cluster')
fig = px.scatter(metadata, x='lon', y='lat',color='cluster')
                    #color='petal_length', symbol='species')

fig.update_traces(marker=dict(size=2),
                  selector=dict(mode='markers'))
fig.write_html("clustered_lakes_2d_conus_"+str(n_clusters)+".html")
fig.show()

