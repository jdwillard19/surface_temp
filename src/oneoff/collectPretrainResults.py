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

best16_paths = np.empty((n_lakes),dtype=np.object)
best32_paths = np.empty((n_lakes),dtype=np.object)
best64_paths = np.empty((n_lakes),dtype=np.object)
best128_paths = np.empty((n_lakes),dtype=np.object)

def job_path(site_id):
    return "../hpc/jobs/train_glm_emulator_"+str(site_id)+".out"
for site_ct, site_id in enumerate(train_ids):
    print("starting lake #",site_ct,": ", site_id)
    #load result
    out_f = open(job_path(site_id), "r")
    out_txt = out_f.read()
    match16 = re.findall(pattern="BEST_MODEL_PATH_16HID:../../models/"+site_id+"/glm_emulator_16hid_(\d+)ep\nMean\s+RMSE=(\d*\.?\d+)",string=out_txt,flags=re.M | re.DOTALL)
    err_per_glm16[site_ct] = match16[0][1]
    best16_paths[site_ct] = "../../models/"+site_id+"/LSTM_source_model_16hid_"+match16[0][0]+"ep"

    match32 = re.findall(pattern="BEST_MODEL_PATH_32HID:../../models/"+site_id+"/glm_emulator_32hid_(\d+)ep\nMean\s+RMSE=(\d*\.?\d+)",string=out_txt,flags=re.M | re.DOTALL)
    err_per_glm32[site_ct] = match32[0][1]
    best32_paths[site_ct] = "../../models/"+site_id+"/LSTM_source_model_32hid_"+match32[0][0]+"ep"


    match64 = re.findall(pattern="BEST_MODEL_PATH_64HID:../../models/"+site_id+"/glm_emulator_64hid_(\d+)ep\nMean\s+RMSE=(\d*\.?\d+)",string=out_txt,flags=re.M | re.DOTALL)
    err_per_glm64[site_ct] = match64[0][1]
    best64_paths[site_ct] = "../../models/"+site_id+"/LSTM_source_model_64hid_"+match64[0][0]+"ep"


    match128 = re.findall(pattern="BEST_MODEL_PATH_128HID:../../models/"+site_id+"/glm_emulator_128hid_(\d+)ep\nMean\s+RMSE=(\d*\.?\d+)",string=out_txt,flags=re.M | re.DOTALL)
    err_per_glm128[site_ct] = match128[0][1]
    best128_paths[site_ct] = "../../models/"+site_id+"/LSTM_source_model_128hid_"+match128[0][0]+"ep"


pdb.set_trace()
min16_ind = np.argmin(err_per_glm16)
min32_ind = np.argmin(err_per_glm32)
min64_ind = np.argmin(err_per_glm64)
min128_ind = np.argmin(err_per_glm128)

site16 = train_ids[min16_ind]
print("best 16: ",best16_paths[min16_ind])
print("RMSE: ", err_per_glm16[min16_ind])

site32 = train_ids[min32_ind]
print("best 32: ",best32_paths[min32_ind])
print("RMSE: ", err_per_glm32[min32_ind])


site64 = train_ids[min64_ind]
print("best 64: ",best64_paths[min64_ind])
print("RMSE: ", err_per_glm64[min64_ind])

site128 = train_ids[min128_ind]
print("best 128: ",best128_paths[min128_ind])
print("RMSE: ", err_per_glm128[min128_ind])
