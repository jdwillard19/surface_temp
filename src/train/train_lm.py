import pandas as pd
import numpy as np
import pdb
import sys
import os
from sklearn.feature_selection import RFECV
from joblib import dump, load
import re
import datetime
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import linear_model
##################################################################3
# (Jan 2020 - Jared) - 
####################################################################3

currentDT = datetime.datetime.now()
print("script start: ",str(currentDT))

#file to save model  to
save_file_path = '../../models/lm_surface_temp.joblib'




#############################
#load data
train_lakes = np.load("../../data/static/lists/source_lakes_wrr.npy")
train_lakes_wp = ["nhdhr_"+x for x in train_lakes]

columns = ['ShortWave_t-30','LongWave_t-30','AirTemp_t-30','WindSpeed_t-30',
           'ShortWave_t-14','LongWave_t-14','AirTemp_t-14','WindSpeed_t-14',\
           'ShortWave_t-4','LongWave_t-4','AirTemp_t-4','WindSpeed_t-4',\
           'ShortWave_t-3','LongWave_t-3','AirTemp_t-3','WindSpeed_t-3',\
           'ShortWave_t-2','LongWave_t-2','AirTemp_t-2','WindSpeed_t-2',\
           'ShortWave_t-1','LongWave_t-1','AirTemp_t-1','WindSpeed_t-1',\
           'ShortWave','LongWave','AirTemp','WindSpeed',\
           'Surface_Area','Surface_Temp']
feat_inds = [0,1,2,4,8]
train_df = pd.DataFrame(columns=columns)

param_search = True

lookback = 4
farthest_lookback = 30
#build training set
for ct, lake_id in enumerate(train_lakes):
    #load data
    feats = np.load("../../data/processed/"+lake_id+"/features_ea.npy")
    labs = np.load("../../data/processed/"+lake_id+"/full.npy")
    # dates = np.load("../../data/processed/"+name+"/dates.npy")
    data = np.concatenate((feats[:,feat_inds],labs.reshape(labs.shape[0],1)),axis=1)
    X = data[:,:-1]
    y = data[:,-1]
    if lookback > 0:
        pdb.set_trace()
        X = np.array([np.append(np.append(np.append(X[i,:],X[i-lookback:i,:4].flatten()),X[i-14,:4]),X[i-30,:4]) for i in np.arange(farthest_lookback,X.shape[0])],dtype = np.half)
        y = y[farthest_lookback:]
    #remove days without obs
    data = np.concatenate((X,y.reshape(len(y),1)),axis=1)

    data = data[np.where(np.isfinite(data[:,-1]))]
    new_df = pd.DataFrame(columns=columns,data=data)
    train_df = pd.concat([train_df, new_df], ignore_index=True)


X = train_df[columns[:-1]].values
y = np.ravel(train_df[columns[-1]].values)
pdb.set_trace()

print("train set dimensions: ",X.shape)
#construct lookback feature set??
if param_search:
  lm = linear_model.LinearRegression()
  rfecv = RFECV(estimator=lm, cv=24, step=2, scoring='neg_mean_squared_error', verbose=1, n_jobs=24)

  # selection = SelectFromModel(gbm, threshold=0.03, prefit=False)
  # selection.fit(X_trn,y_trn)
  # selected_dataset = selection.transform(X_tst)
  rfecv.fit(X, y)

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


pdb.set_trace()
#create and fit model
model = linear_model.LinearRegression()


# cv = cross_val_score(model, X, y=y, cv=12, n_jobs=12, verbose=1)
# print("cv scores ", cv)
# print(np.mean(cv))
# sys.exit()
# pdb.set_trace()
print("Training XGB regression model...")
model.fit(X, y)
dump(model, save_file_path)
print("model trained and saved to ", save_file_path)





