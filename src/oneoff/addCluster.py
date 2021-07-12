import pandas as pd
import pdb
df_dr = pd.read_csv("../../../lake_conus_surface_temp_2021/metadata/lake_metadata.csv")
df_old = pd.read_csv("../../metadata/surface_lake_metadata_041421_wCluster.csv")
df_old.set_index('site_id',inplace=True)
cluster = []

ct = 0
for site_id in df_dr['site_id'].values:
	ct+=1
	print(ct)
	if site_id not in df_old.index:
		cluster.append(None)
	else:
		cluster.append(df_old.loc[site_id]['cluster'])

pdb.set_trace()

df_dr['cluster'] = cluster
