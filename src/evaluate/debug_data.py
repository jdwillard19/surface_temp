import pandas as pd
import numpy as np
import sys
import os
import pdb
import matplotlib.pyplot as plt





dates = np.load("debug_dates.npy")
inputs = np.load("debug_inputs.npy")
outputs = np.load("debug_predictions.npy")
targets = np.load("debug_targets.npy")
trn_data = np.load("ealstm_trn_data_final_041321.npy")
trn_data = trn_data[:,:,:-1]
dates_b = dates[50]
inputs_b = inputs[50]
outputs_b = outputs[50]

dates_g = dates[31]
inputs_g = inputs[31]
outputs_g = outputs[31]

start_ind_b = 0
start_ind_g = 38
end_ind_b = -39
end_ind_g = -1

dates_b = dates_b[start_ind_b:end_ind_b]
dates_g = dates_g[start_ind_g:end_ind_g]

inputs_b = inputs_b[start_ind_b:end_ind_b]
inputs_g = inputs_g[start_ind_g:end_ind_g]

outputs_b = outputs_b[start_ind_b:end_ind_b]
outputs_g = outputs_g[start_ind_g:end_ind_g]


#DECEMBER THRU OCTOBER
pdb.set_trace()
xarr = np.arange(dates_b.shape[0])

#plot outputs
# plt.plot(xarr,outputs_b,color='red')
# plt.plot(xarr,outputs_g,color='blue')
# plt.show()

#plot inputs
plt.plot(xarr,inputs_b[:,5],color='red')
plt.plot(xarr,inputs_g[:,5],color='blue')
plt.show()