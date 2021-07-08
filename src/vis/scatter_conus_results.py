import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
data = pd.read_csv("../../results/all_test_obs_model_vs_actual_w_date.csv")
data2 = pd.read_csv("../../results/err_and_obs_per_test_site.csv")

out = data['outputs'].values
lab = data['labels'].values


#scatterplot

# plt.scatter(out, lab,alpha=0.02,color='black', label='Target Lakes (Test)', s=5)
# plt.xlabel("EA-LSTM Output")
# plt.ylabel("Measured Lake Surface Temp")
# line_data = np.linspace(0,40,700)
# plt.plot(line_data, line_data, color='red',linewidth=1)
# plt.show()


#bean plot
# pdb.set_trace()
# ax = sns.violinplot(y="rmse", data=data2, palette="muted", scale='width')
# ax.set_autoscale_on(False)
# for i in range(90):

# 	plt.plot(['GLM', 'PGMTL'], [glm[i], pg[i]], color='black', linewidth=0.5)
	# plt.plot(['PGMTL_50','PGMTL_100', 'PGMTL_150', 'PGMTL_200', 'PGMTL_all', 'GLM'], [glm[i], y_full[i], y_200[i], y_150[i], y_100[i], y_50[i]], color='black', linewidth=1)
plt.show()