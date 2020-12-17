import pandas as pd
import numpy as np
import re
import pdb


train_lakes = np.load("../../data/static/lists/source_lakes_wrr.npy")
test_lakes = np.load("../../data/static/lists/target_lakes_wrr.npy",allow_pickle=True)
train_lakes = test_lakes
train_lakes_wp = ["nhdhr_"+x for x in train_lakes]

train_df = pd.DataFrame()
train_df2 = pd.DataFrame()
for lake_id in train_lakes:
    pdb.set_trace()
    # lake_df_res = pd.read_csv("../../results/transfer_learning/target_"+lake_id+"/PGDL_transfer_results",header=None,names=['source_id','rmse'])
    lake_df_res = pd.read_csv("../../results/transfer_learning/target_"+lake_id+"/PGDL_transfer_results_targets",header=None,names=['source_id','rmse'])
    # lake_df_res = pd.read_csv("../../results/transfer_learning/target_"+lake_id+"/PGDL_transfer_results_noTran_noPre",header=None,names=['source_id','rmse'])
    lake_df_res2 = pd.read_csv("../../results/transfer_learning/target_"+lake_id+"/PGDL_transfer_results_noTran_wPre",header=None,names=['source_id','rmse'])
    lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']
    lake_df_res2 = lake_df_res2[lake_df_res2.source_id != 'source_id']

    #get metadata differences between target and all the sources
    # lake_df = pd.read_feather("../../metadata/diffs/target_nhdhr_"+lake_id+".feather")
    # lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes_wp)]
    # lake_df2 = lake_df[:]
    lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
    lake_df_res2 = lake_df_res2[np.isin(lake_df_res2['source_id'], train_lakes)]
    lake_df_res['source_id2'] = ['nhdhr_'+str(x) for x in lake_df_res['source_id'].values]
    lake_df_res2['source_id2'] = ['nhdhr_'+str(x) for x in lake_df_res2['source_id'].values]
    # lake_df = pd.merge(left=lake_df, right=lake_df_res.astype('object'), left_on='site_id', right_on='source_id2')
    # lake_df2 = pd.merge(left=lake_df2, right=lake_df_res2.astype('object'), left_on='site_id', right_on='source_id2')
    # new_df = lake_df
    new_df = lake_df_res
    new_df2 = lake_df_res2
    train_df = pd.concat([train_df, new_df], ignore_index=True)
    train_df2 = pd.concat([train_df2, new_df2], ignore_index=True)


pdb.set_trace()
