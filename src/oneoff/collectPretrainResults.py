import pandas as pd
import numpy as np
import re
import pdb


train_ids = np.load("../../data/static/lists/source_lakes_wrr")

job_base_path = "../hpc/jobs/train_glm_emulator_"

def job_path(site_id):
	return "../hpc/jobs/train_glm_emulator_"+str(site_id)+".out"
for site_id in train_ids:
	#load result
    out_f = open(, "r")
    out_txt = out_f.read(job_path(site_id))
    match16 = re.findall(pattern="BEST_MODEL_PATH_16HID:../../models/105954753/LSTM_source_model_16hid_(\d+)ep\nRMSE=(\d*\.?\d+)",string=out_txt,flags=flags=re.M | re.DOTALL)
    pdb.set_trace()
    hid16rmse = match16[2]