import pandas as pd
import numpy as np
import pdb
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFECV
import math
import re
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel

###################################################################################################
# (Sept 2020 - Jared) - This script performs feature selection for PG-MTL metamodel
##############################################################################################


#load source lake list
# glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
# train_df = pd.read_feather("../../results/transfer_learning/glm/train_rmses_pball.feather")
# train_lakes = [re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)]
train_lakes = np.load("../../data/static/lists/source_lakes_wrr.npy")
train_lakes_wp = ["nhdhr_"+x for x in train_lakes]
# test_lakes = np.load("../../data/static/lists/target_lakes_wrr.npy",allow_pickle=True)

# train_lakes_wp = np.unique(glm_all_f['target_id'].values)
n_lakes = len(train_lakes)
# feats = train_df.columns[80:-1]

train_df = pd.DataFrame()
#PGDL_transfer_results_noTran_wPre
feats = pd.read_feather("../../metadata/diffs/target_nhdhr_91685677.feather").columns[1:][75:]


#compile all the meta-features and meta-target values into one dataframe for training
for _, lake_id in enumerate(train_lakes):

	new_df = pd.DataFrame()

	#get performance results (metatargets), filter out target as source
	lake_df_res = pd.read_csv("../../results/transfer_learning/target_"+lake_id+"/PGDL_transfer_results_noTran_wPre",header=None,names=['source_id','rmse'])
	lake_df_res = lake_df_res[lake_df_res.source_id != 'source_id']

	#get metadata differences between target and all the sources
	lake_df = pd.read_feather("../../metadata/diffs/target_nhdhr_"+lake_id+".feather")
	lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes_wp)]
	lake_df_res = lake_df_res[np.isin(lake_df_res['source_id'], train_lakes)]
	lake_df_res['source_id2'] = ['nhdhr_'+str(x) for x in lake_df_res['source_id'].values]
	lake_df = pd.merge(left=lake_df, right=lake_df_res.astype('object'), left_on='site_id', right_on='source_id2')
	new_df = lake_df
	train_df = pd.concat([train_df, new_df], ignore_index=True)

print("training dataset size, ",train_df.shape[0],"*",train_df.shape[1])
# test_df = pd.DataFrame()

# for targ_ct, target_id in enumerate(test_lakes): #for each target lake
#     # print(str(targ_ct),'/',len(test_lakes),':',target_id)
#     lake_df = pd.DataFrame()
#     lake_id = target_id
#     lake_df = pd.read_feather("../../metadata/diffs/target_nhdhr_"+lake_id+".feather")
#     lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes_wp)]
#     X = pd.DataFrame(lake_df[feats])
#     test_df = pd.concat([test_df, lake_df])


#declare model and predictors and response
# est = GradientBoostingRegressor(n_estimators=800, learning_rate=0.1)
X_trn = pd.DataFrame(train_df[feats])
# X_tst = pd.DataFrame(test_df[feats])
y_trn = np.ravel(pd.DataFrame(train_df['rmse']))
# y_tst = np.ravel(pd.DataFrame(test_df['rmse']))
dtrain = xgb.DMatrix(X_trn, label=y_trn)

#perform recursive feature elimination
gbm = xgb.XGBRegressor(booster='gbtree')
rfecv = RFECV(estimator=gbm, cv=24, step=2, scoring='neg_mean_squared_error', verbose=1, n_jobs=29)

# selection = SelectFromModel(gbm, threshold=0.03, prefit=False)
# selection.fit(X_trn,y_trn)
# selected_dataset = selection.transform(X_tst)
rfecv.fit(X_trn, y_trn)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
print("ranking: ", rfecv.ranking_)

# print("scores: ", selection.estimator_.coef_)
# print("support: ",selection.get_support())
# new_feats = feats[selection.get_support()]
# X_trn = pd.DataFrame(train_df[new_feats])
# y_trn = np.ravel(pd.DataFrame(train_df['rmse']))
# # y_tst = np.ravel(pd.DataFrame(test_df['rmse']))
# dtrain = xgb.DMatrix(X_trn, label=y_trn)

# print("new feats: ", new_feats)
# print(selection.estimator_.feature_importances_[selection.get_support()])
# X_trn2 = pd.DataFrame(train_df[new_feats])
# X_tst2 = pd.DataFrame(test_df[feats])
# gbm = xgb.XGBRegressor(booster='gbtree').fit(X_trn2, X_tst2)
# cv  = xgb.cv(data = dtrain, nrounds = 3, nthread = -1, nfold = 12, metrics = list("rmse"),
#                   max_depth = 3, eta = 1, objective = "binary:logistic")
# print(cv)

# pdb.set_trace()
print("ranking: ", repr(rfecv.ranking_))

print("scores: ", repr(rfecv.grid_scores_))

print("selected features\n---------------------------------------------------------------------------")
# print(feats[selector.get_support()])
print(feats[rfecv.ranking_==1])
print("------------------------------------------------------------------------")





