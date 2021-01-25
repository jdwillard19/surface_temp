import xarray as xr
import pandas as pd
import feather
import numpy as np
import os
import sys
import re
import math
import shutil
from scipy import interpolate
import pdb
import datetime



#load metadata, get ids
metadata = pd.read_csv("../../metadata/surface_lake_metadata_conus.csv")
pdb.set_trace()
#load wst obs
obs = pd.read_feather("../../data/raw/obs/temp_wqp_munged.feather")


#load NLDAS data
lw_ds_path = "../../data/globus/NLDAS_DLWRFsfc_19790102-20210102_train_test.nc" #longwave
at_ds_path = "../../data/globus/NLDAS_TMP2m_19790102-20210102_train_test.nc" #airtemp
sw_ds_path = "../../data/globus/NLDAS_DSWRFsfc_19790102-20210102_train_test.nc" #shortwave
#wsu_ds_path=

lw_ds = xr.open_dataset(lw_ds_path)
at_ds = xr.open_dataset(at_ds_path)
sw_ds = xr.open_dataset(sw_ds_path)


