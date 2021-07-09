from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.init import xavier_normal_
from datetime import date
import pandas as pd
import pdb
import random
import math
import sys
import re
import os
sys.path.append('../../data')
sys.path.append('../data')
sys.path.append('../../models')
sys.path.append('../models')
from pytorch_model_operations import saveModel
import pytorch_data_operations
import datetime
from torch.utils.data import DataLoader
from pytorch_data_operations import buildLakeDataForRNN_repr_trn, buildLakeDataForRNN_repr_tst


#get site Iids
site_ids = np.load("../../metadata/lakeset.npy",allow_pickle=True)

(trn_data, trn_dates) = buildLakeDataForRNN_repr_trn(site_ids) 


# trn_path = 
# if not os.path.exists("./ctlstm_trn_data.npy"):
#     np.save("./ealstm_trn_data_062321.npy",trn_data)
# else:
#     trn_data = torch.from_numpy(np.load("./ealstm_trn_data_062321.npy"))
