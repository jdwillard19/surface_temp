import pandas as pd
import numpy as np
import re
import pdb


train_ids = np.load("../../data/static/lists/source_lakes_wrr.npy")
n_lakes = train_ids.shape[0]
job_base_path = "../hpc/jobs/train_glm_emulator_"

err_per_glm16 = np.empty((n_lakes))
err_per_glm16[:] = np.nan

err_per_glm32 = np.empty((n_lakes))
err_per_glm32[:] = np.nan

err_per_glm64 = np.empty((n_lakes))
err_per_glm64[:] = np.nan

err_per_glm128 = np.empty((n_lakes))
err_per_glm128[:] = np.nan

def job_path(site_id):
    return "../hpc/jobs/train_glm_emulator_"+str(site_id)+".out"
for site_ct, site_id in enumerate(train_ids):
    print("starting lake #",site_ct,": ", site_id)
    #load result
    out_f = open(job_path(site_id), "r")
    out_txt = out_f.read()
    match16 = re.findall(pattern="BEST_MODEL_PATH_16HID:../../models/"+site_id+"/glm_emulator_16hid_(\d+)ep\nMean\s+RMSE=(\d*\.?\d+)",string=out_txt,flags=re.M | re.DOTALL)
    err_per_glm16[site_ct] = match16[0][1]

    match32 = re.findall(pattern="BEST_MODEL_PATH_32HID:../../models/"+site_id+"/glm_emulator_32hid_(\d+)ep\nMean\s+RMSE=(\d*\.?\d+)",string=out_txt,flags=re.M | re.DOTALL)
    err_per_glm32[site_ct] = match32[0][1]

    match64 = re.findall(pattern="BEST_MODEL_PATH_64HID:../../models/"+site_id+"/glm_emulator_64hid_(\d+)ep\nMean\s+RMSE=(\d*\.?\d+)",string=out_txt,flags=re.M | re.DOTALL)
    err_per_glm64[site_ct] = match64[0][1]

    match128 = re.findall(pattern="BEST_MODEL_PATH_128HID:../../models/"+site_id+"/glm_emulator_128hid_(\d+)ep\nMean\s+RMSE=(\d*\.?\d+)",string=out_txt,flags=re.M | re.DOTALL)
    err_per_glm128[site_ct] = match128[0][1]


pdb.set_trace()