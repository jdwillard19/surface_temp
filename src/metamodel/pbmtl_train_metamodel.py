import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump, load

################################################################################3
# (Sept 2020 - Jared) - train PB-MTL model on 145 source lake set
# Features and hyperparamters must be manually specified below 
# (e.g. feats = ['dif_max_depth', ....]; n_estimators = 4000, etc)
#############################################################################################################

#file to save model  to
save_file_path = '../../models/metamodel_glm_RMSE_GBR.joblib'


#load training data
train_df = pd.read_feather("../../results/glm_transfer/glm_meta_train_data.feather")



#########################################################################################
#paste features found in "pbmtl_feature_selection.py" here
feats = ['n_obs_su', 'obs_temp_mean', 'obs_temp_skew', 'obs_temp_kurt',
       'ad_glm_strat_perc', 'obs_temp_mean_airdif', 'dif_max_depth',
       'dif_surface_area', 'dif_sw_mean_au', 'dif_ws_mean_au',
       'dif_lathrop_strat', 'dif_glm_strat_perc', 'perc_dif_max_depth',
       'perc_dif_sqrt_surface_area']
feats = 
###################################################################################

#######################################################################3
#paste hyperparameters found in "pbmtl_hyperparameter_search.py" here
#
n_estimators = 300
objective = 'reg:squarederror'
learning_rate = .05
colsample_bytree=.7
max_depth=6
min_child_weight=11
subsample=.8
#####################################################################


#{'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 11, 'n_estimators': 300, 'objective': 'reg:squarederror', 'subsample': 0.8}

########################
##########################
#metamodel training code
##########################
#######################


print("Model training in progress...")
model = xgb.XGBRegressor(booster='gbtree',n_estimators=n_estimators,objective=objective,learning_rate=learning_rate,colsample_bytree=colsample_bytree,
						 max_depth=6,min_child_weight=min_child_weight,subsample=subsample)

X = pd.DataFrame(train_df[feats])
y = np.ravel(pd.DataFrame(train_df['rmse']))

model.fit(X,y)
dump(model, save_file_path) 
print("model train complete, saved to ", save_file_path)


