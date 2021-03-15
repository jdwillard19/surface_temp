import pandas as pd
import os




metadata = pd.read_csv("../../metadata/lake_metadata_full_conus_185k.csv")
test_lakes = metadata['site_id'].values
for i, target_id in enumerate(test_lakes):
    if target_id == "nhdhr_{ef5a02dc-f608-4740-ab0e-de374bf6471c}" or target_id == 'nhdhr_136665792' or target_id == 'nhdhr_136686179':
        continue

    print(i,"/",len(test_lakes))
    df = pd.read_feather('../../results/SWT_results/outputs_'+target_id+'.feather')
    assert df.shape[0] == 14976, "output not correct size"