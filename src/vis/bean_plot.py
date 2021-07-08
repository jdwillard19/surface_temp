import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb

data = pd.read_feather("../../results/err_per_site_031821.feather")
# y_pg = pd.read_csv("pgdtl_glm_testrmses.csv", header=None)
# pdb.set_trace()
ea = data['rmse_ealstm'].values.tolist()
gb = data['rmse_xgboost'].values.tolist()
lr = data['rmse_lm'].values.tolist()
# pg = y_pg[1].tolist()
# glm = y_pg[2].tolist()
ea_x = ['EA-LSTM'] * len(ea)
gb_x = ['XGBoost'] * len(gb)
lm_x = ['Linear Regression'] * len(lr)
x = ea_x + gb_x + lm_x
y = ea + gb + lr


df = pd.DataFrame()
# df2 = pd.DataFrame()
# x.reverse()
# x_sp.reverse()
df['Method'] = x
df['RMSE (degrees C)'] = y  
# df2['Method'] = x_sp
# df2['RMSE (degrees C)'] = glm + y_full + y_200 + y_150 + y_100+ y_50   
ax = sns.violinplot(x="Method", y="RMSE (degrees C)", data=df, palette="muted", scale='width')
ax.set_autoscale_on(False)
ax.set_ylim([0, 8])
# for i in range(90):

# 	plt.plot(['GLM', 'PGMTL'], [glm[i], pg[i]], color='black', linewidth=0.5)
	# plt.plot(['PGMTL_50','PGMTL_100', 'PGMTL_150', 'PGMTL_200', 'PGMTL_all', 'GLM'], [glm[i], y_full[i], y_200[i], y_150[i], y_100[i], y_50[i]], color='black', linewidth=1)
plt.show()