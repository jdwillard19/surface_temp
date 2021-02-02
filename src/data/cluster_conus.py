import pandas as pd
import numpy as np
import pdb
import sys
import os
# import plotly.express as px
from sklearn.cluster import KMeans

metadata = pd.read_csv("../../metadata/surface_lake_metadata_conus.csv")
metadata = pd.read_csv("../../metadata/surface_lake_metadata_file_temp.csv")
data2 = pd.read_csv("../../results/err_and_obs_per_test_site.csv")

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

cluster_label_vals = np.unique(kmeans.labels_)

# metadata['train'] = True
# for i in cluster_label_vals:
# 	test_inds = np.random.choice(np.where(metadata['cluster'] ==i)[0],size=int(np.round(np.where(metadata['cluster']==i)[0].shape[0]/3)))
# 	metadata.iloc[test_inds, metadata.columns.get_loc('train')] = False


#lad train/test
source_lakes = np.load("../../data/static/lists/source_lakes_conus.npy",allow_pickle=True)
target_lakes = np.load("../../data/static/lists/test_lakes_conus.npy",allow_pickle=True)
source_meta = metadata[np.isin(metadata,source_lakes)]
target_meta = metadata[np.isin(metadata,target_lakes)]
target_meta = target_meta[np.isin(target_meta['site_id'],data2['site_id'])]
target_meta.reset_index(inplace=True)
target_meta['rmse'] = data2['rmse']
rmse_per_cluster = [np.median(target_meta[target_meta['cluster']==i]['rmse']) for i in cluster_label_vals]
err_per_clust = pd.DataFrame()
err_per_clust['cluster'] = cluster_label_vals
err_per_clust['Median RMSE'] = rmse_per_cluster
target_meta.to_csv("../../metadata/conus_target_meta.csv")
souce_meta.to_csv("../../metadata/conus_source_meta.csv")
sys.exit()
#declare train/test
pdb.set_trace()
# fig = px.scatter_3d(metadata, x='lon', y='lat', z='log_area',color='cluster')
# fig = px.scatter_3d(target_meta, x='lon', y='lat', z='rmse',color='cluster')
# for i in cluster_label_vals:
# 	target_meta['isCluster'+str(i)] = (target_meta['cluster'] == i)
# 	fig = px.scatter(target_meta, x='lon', y='lat',color='isCluster'+str(i))
# 	# fig = px.scatter(metadata[metadata['cluster']==i], x='lon', y='lat',color='cluster')
# 	# fig = px.scatter(metadata[metadata['cluster'] != i], x='lon', y='lat')
# 	fig.update_traces(marker=dict(size=5),
#                   selector=dict(mode='markers'))
# 	fig.write_html("clustered_lakes-conus_cluster_"+str(i)+".html")

#                     #color='petal_length', symbol='species')

# sys.exit()
# fig.write_html("clustered_lakes-conus_"+str(n_clusters)+"clusters_rmse_z_axis.html")
# fig.show()


