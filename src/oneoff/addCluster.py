import pandas as pd
df_dr = pd.read_csv("../../../lake_conus_surface_temp_2021/metadata/lake_metadata.csv")
df_old = pd.read_csv("../../metadata/surface_lake_metadata_041421_wCluster.csv")

cluster = []

ct = 0
for site_id in df_dr['site_id'].values:
	ct+=1
	print(ct)
	if df_old[df_old['site_id']==site_id]['cluster'].values.shape[0] == 0:
		cluster.append(None)
	else:
		df_old[df_old['site_id']==site_id]['cluster'].values[0]

pdb.set_trace()

df_dr['cluster'] = cluster
