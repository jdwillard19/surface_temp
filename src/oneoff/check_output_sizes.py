import pandas as pd
import os




metadata = pd.read_csv("../../metadata/lake_metadata_full_conus_185k.csv")
test_lakes = metadata['site_id'].values
for i, target_id in enumerate(test_lakes):
	print(i,"/",len(test_lakes))
	df = pd.read_feather('outputs_'+target_id+'.feather')
	assert df.shape[0] == 14976, "output not correct size"