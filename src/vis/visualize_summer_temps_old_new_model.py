import pdb
import matplotlib.pyplot as plt
import numpy as pd
import pandas as pd
import os
import sys



lakes = ['nhdhr_109986954','nhdhr_109991096','nhdhr_90567371','nhdhr_152287272','nhdhr_154314752']

regexs = ['198[0-4]','198[5-9]',
		  '199[0-4]','199[5-9]',
		  '200[0-4]','200[5-9]',
		  '201[0-4]','201[5-9]','2020']
for site_id in lakes:
	print("site "+site_id)
	old_out = pd.read_feather("../../data/vis/cold_debug/outputs_"+site_id+".feather")
	new_out = pd.read_feather("../../data/vis/cold_debug/outputs_"+site_id+"_COLD_DEBUG.feather")

	for regex in regexs:
		print('regex: ',regex)
		seq1 = old_out[old_out['index'].str.contains(regex)]
		seq2 = new_out[new_out['index'].str.contains(regex)]
		x = seq1['index'].values
		y1 = seq1['temp_pred'].values
		y2 = seq2['temp_pred'].values
		plt.plot(x,y1,color='blue',label='no data filtering')
		plt.plot(x,y2,color='red',label='removed summer cold temps')
		plt.title('Lake: '+site_id+" from "+str(x[0])+ " to "+str(x[-1]))
		plt.ylabel('Deg C')
		plt.savefig("./cold_summer_temp_fix/"+site_id+"_"+str(x[0])+'_'+str(x[-1]))
		plt.clf()

