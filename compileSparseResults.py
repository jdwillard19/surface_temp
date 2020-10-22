
import pandas as pd
import re
import numpy as np
import pdb
import sys
import os
metadata = pd.read_feather("./metadata/lake_metadata_full.feather")
metadata.set_index('site_id', inplace=True)
glm_all_f = pd.read_csv("./results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
train_lakes_wp = np.unique(glm_all_f['target_id'].values) #with prefix

ids = pd.read_csv('./metadata/pball_site_ids.csv', header=None)
ids = ids[0].values
n_lakes = len(train_lakes)
test_lakes = ids[~np.isin(ids, train_lakes)]
pdb.set_trace()
li = []

for site_id in test_lakes:
    filename = "./results/"+site_id+"/sparseModelResults.csv"
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)

frame.to_csv("all_sparse_results.csv")