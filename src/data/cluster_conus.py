import pandas as pd
import numpy as np
import pdb
import sys
import os

metadata = pd.read_csv("../../metadata/surface_lake_metadata_conus.csv")
obs = pd.read_feather("../../data/raw/obs/temp_wqp_munged.feather")


