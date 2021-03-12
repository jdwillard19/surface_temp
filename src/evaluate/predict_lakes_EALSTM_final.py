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
import pdb
from torch.utils.data import DataLoader
from pytorch_data_operations import buildLakeDataForRNN_multilakemodel_conus, parseMatricesFromSeqs, buildLakeDataForRNN_conus, buildLakeDataForRNN_conus_NoLabel



#script start
currentDT = datetime.datetime.now()
print(str(currentDT))

#../../metadata/conus_source_metadata.csv
####################################################3
# (Nov 2020 - Jared) source model script, takes lakename as required command line argument
###################################################33

#enable/disable cuda 
use_gpu = True 
torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=10)



### debug tools
verbose = True
save = True
test = True



start = int(sys.argv[1])
end = int(sys.argv[2])


#####################3
#params
###########################33
first_save_epoch = 0
patience = 200

#ow
seq_length = 350 #how long of sequences to use in model
begin_loss_ind = 0#index in sequence where we begin to calculate error or predict
n_features = 5  #number of physical drivers
n_static_feats = 4
n_total_feats =n_static_feats+n_features
win_shift = 175 #how much to slide the window on training set each time
save = True 
grad_clip = 1.0 #how much to clip the gradient 2-norm in training
dropout = 0.
num_layers = 1
n_hidden = 128
# lambda1 = 1e-
lambda1 = 0

# n_eps = 10000
n_eps = 800
targ_ep = 340
targ_rmse = 2.36
# targ_ep = 0 #DEBUG VALUE
# targ_rmse = 3.5 #DEBUG VALUE

# metadata = pd.read_csv("../../metadata/surface_lake_metadata_021521_wCluster.csv")
metadata = pd.read_csv("../../metadata/lake_metadata_full_conus_185k.csv")
test_lakes = metadata['site_id'].values[start:end]

# metadata = metadata.iloc[150:350] #DEBUG VALUE
obs = pd.read_feather("../../data/raw/obs/surface_lake_temp_daily_020421.feather")

###############################
# data preprocess
##################################
#create train and test sets




#####################################################################################
####################################################3
# fine tune
###################################################33
##########################################################################################33

#####################
#params
###########################
first_save_epoch = 0
epoch_since_best = 0
yhat_batch_size = 1

###############################
# data preprocess
##################################
#create train and test sets





#define EA-LSTM class
"""
This code block is part of the accompanying code to the manuscript:
Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)
You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""


#define LSTM model class
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(myLSTM_Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size = n_total_feats, hidden_size=hidden_size, batch_first=True,num_layers=num_layers,dropout=dropout) #batch_first=True?
        self.out = nn.Linear(hidden_size, 1) #1?
        self.hidden = self.init_hidden()
        self.w_upper_to_lower = []
        self.w_lower_to_upper = []

    def init_hidden(self, batch_size=0):
        # initialize both hidden layers
        if batch_size == 0:
            batch_size = self.batch_size
        ret = (xavier_normal_(torch.empty(num_layers, batch_size, self.hidden_size)),
                xavier_normal_(torch.empty(num_layers, batch_size, self.hidden_size)))
        if use_gpu:
            item0 = ret[0].cuda(non_blocking=True)
            item1 = ret[1].cuda(non_blocking=True)
            ret = (item0,item1)
        return ret

    def forward(self, x, hidden):
        self.lstm.flatten_parameters()
        x = x.float()
        x, hidden = self.lstm(x, self.hidden)
        self.hidden = hidden
        x = self.out(x)
        return x, hidden

#method to calculate l1 norm of model
def calculate_l1_loss(model):
    def l1_loss(x):
        return torch.abs(x).sum()

    to_regularize = []
    # for name, p in model.named_parameters():
    for name, p in model.named_parameters():
        if 'bias' in name:
            continue
        else:
            #take absolute value of weights and sum
            to_regularize.append(p.view(-1))
    l1_loss_val = torch.tensor(1, requires_grad=True, dtype=torch.float32)
    l1_loss_val = l1_loss(torch.cat(to_regularize))
    return l1_loss_val


class EALSTM(nn.Module):
    """Implementation of the Entity-Aware-LSTM (EA-LSTM)
    TODO: Include paper ref and latex equations
    Parameters
    ----------
    input_size_dyn : int
        Number of dynamic features, which are those, passed to the LSTM at each time step.
    input_size_stat : int
        Number of static features, which are those that are used to modulate the input gate.
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        super(EALSTM, self).__init__()

        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # create tensors of learnable parameters
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size_dyn, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.weight_sh = nn.Parameter(torch.FloatTensor(input_size_stat, hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        self.bias_s = nn.Parameter(torch.FloatTensor(hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.weight_sh)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, val=0)
        nn.init.constant_(self.bias_s.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x_d, x_s):
        """[summary]
        Parameters
        ----------
        x_d : torch.Tensor
            Tensor, containing a batch of sequences of the dynamic features. Shape has to match
            the format specified with batch_first.
        x_s : torch.Tensor
            Tensor, containing a batch of static features.
        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor]
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x_d = x_d.transpose(0, 1)
            # x_s = x_s.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []
        # expand bias vectors to batch size
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        # calculate input gate only once because inputs are static
        bias_s_batch = (self.bias_s.unsqueeze(0).expand(batch_size, *self.bias_s.size()))

        i = torch.sigmoid(torch.addmm(bias_s_batch, x_s, self.weight_sh))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh) +
                     torch.mm(x_d[t], self.weight_ih))
            f, o, g = gates.chunk(3, 1)

            c_1 = torch.sigmoid(f) * c_0 + i * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n

class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connceted layer"""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0,
                 concat_static: bool = False,
                 no_static: bool = False):
        """Initialize model.
        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        input_size_stat: int
            Number of static input features (used in the EA-LSTM input gate).
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        concat_static: bool
            If True, uses standard LSTM otherwise uses EA-LSTM
        no_static: bool
            If True, runs standard LSTM
        """
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static

        if self.concat_static or self.no_static:
            self.lstm = LSTM(input_size=input_size_dyn,
                             hidden_size=hidden_size,
                             initial_forget_bias=initial_forget_bias)
        else:
            self.lstm = EALSTM(input_size_dyn=input_size_dyn,
                               input_size_stat=input_size_stat,
                               hidden_size=hidden_size,
                               initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.
        Parameters
        ----------
        x_d : torch.Tensor
            Tensor containing the dynamic input features of shape [batch, seq_length, n_features]
        x_s : torch.Tensor, optional
            Tensor containing the static catchment characteristics, by default None
        Returns
        -------
        out : torch.Tensor
            Tensor containing the network predictions
        h_n : torch.Tensor
            Tensor containing the hidden states of each time step
        c_n : torch,Tensor
            Tensor containing the cell states of each time step
        """
        if self.concat_static or self.no_static:
            h_n, c_n = self.lstm(x_d)
        else:
            h_n, c_n = self.lstm(x_d, x_s)
        h_n = self.dropout(h_n)
        # last_h = self.dropout(h_n[:, -1, :])
        out = self.fc(h_n)
        return out, h_n, c_n

    # © 2020 GitHub, Inc.
    # Terms
    # Privacy
    # Security
    # Status
    # Help

    # Contact GitHub
    # Pricing
    # API
    # Training
    # Blog
    # About




# lstm_net = myLSTM_Net(n_total_feats, n_hidden, batch_size)
lstm_net = Model(input_size_dyn=n_features,input_size_stat=n_static_feats,hidden_size=n_hidden)

#tell model to use GPU if needed
if use_gpu:
    lstm_net = lstm_net.cuda()

load_path = "../../models/EALSTM_256hid_1_final"
n_hidden = torch.load(load_path)['state_dict']['lstm.weight_hh'].shape[0]
lstm_net = Model(input_size_dyn=n_features,input_size_stat=n_static_feats,hidden_size=n_hidden)
if use_gpu:
    lstm_net = lstm_net.cuda(0)
pretrain_dict = torch.load(load_path)['state_dict']
model_dict = lstm_net.state_dict()
pretrain_dict = {key: v for key, v in pretrain_dict.items() if key in model_dict}
model_dict.update(pretrain_dict)
lstm_net.load_state_dict(pretrain_dict)


mse_criterion = nn.MSELoss()


#after training, do test predictions / error estimation
for targ_ct, target_id in enumerate(test_lakes): #for each target lake
    if target_id == "nhdhr_{ef5a02dc-f608-4740-ab0e-de374bf6471c}" or target_id == 'nhdhr_136665792' or target_id == 'nhdhr_136686179':
        continue



    print(str(targ_ct),'/',len(test_lakes),':',target_id)
    lake_df = pd.DataFrame()
    lake_id = target_id

    # lake_df = pd.read_feather("../../metadata/diffs/target_nhdhr_"+lake_id+".feather")
    # lake_df = lake_df[np.isin(lake_df['site_id'], train_lakes_wp)]
    # X = pd.DataFrame(lake_df[feats])


    # y_pred = model.predict(X)
    # lake_df['rmse_pred'] = y_pred

    # lake_df.sort_values(by=['rmse_pred'], inplace=True)
    # lowest_rmse = lake_df.iloc[0]['rmse_pred']
# 
    # top_ids = [str(j) for j in lake_df.iloc[:k]['site_id']]
    
    # best_site = top_ids[0]




    data_dir_target = "../../data/processed/"+target_id+"/" 
    #target agnostic model and data params
    use_gpu = True
    # n_hidden = 20
    seq_length = 350
    win_shift = 175
    begin_loss_ind = 0

    observed = False
    if metadata[metadata['site_id']==target_id]['observed'].values[0]:
        (tst_data_target, tst_dates) = buildLakeDataForRNN_conus(target_id, data_dir_target, seq_length, n_features,
                                       win_shift = win_shift, begin_loss_ind = begin_loss_ind, 
                                       outputFullTestMatrix=True, allTestSeq=True, n_static_feats=n_static_feats)
    else:
        (tst_data_target, tst_dates) = buildLakeDataForRNN_conus_NoLabel(target_id, data_dir_target, seq_length, n_features,
                                       win_shift = win_shift, begin_loss_ind = begin_loss_ind, 
                                       outputFullTestMatrix=True, allTestSeq=True, n_static_feats=n_static_feats)
    unique_tst_dates_target = np.unique(tst_dates)
    #useful values, LSTM params
    batch_size = tst_data_target.size()[0]
    n_test_dates_target = unique_tst_dates_target.shape[0]

    testloader = torch.utils.data.DataLoader(tst_data_target, batch_size=tst_data_target.size()[0], shuffle=False, pin_memory=True)

    with torch.no_grad():
        avg_mse = 0
        ct = 0
        for m, data in enumerate(testloader, 0):
            #now for mendota data
            #this loop is dated, there is now only one item in testloader

            #parse data into inputs and targets
            inputs = data[:,:,:n_total_feats].float()
            targets = data[:,:,-1].float()
            targets = targets[:, begin_loss_ind:]
            tmp_dates = tst_dates[:, begin_loss_ind:]

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            #run model
            h_state = None
            # lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            # outputs, h_state, c_state = lstm_net(inputs[:,:,:n_features], inputs[:,0,n_features:])
            pred, h_state, _ = lstm_net(inputs[:,:,n_static_feats:], inputs[:,0,:n_static_feats])
            pred = pred.view(pred.size()[0],-1)
            pred = pred[:, begin_loss_ind:]

            #calculate error
            targets = targets.cpu()
            loss_indices = np.where(np.isfinite(targets))
            if use_gpu:
                targets = targets.cuda()
            inputs = inputs[:, begin_loss_ind:, :]
            mse = mse_criterion(pred[loss_indices], targets[loss_indices])
            # print("test loss = ",mse)
            avg_mse += mse
            ct += 1
            # if mse > 0: #obsolete i think
            #     ct += 1
        avg_mse = avg_mse / ct

        (outputm_npy, labelm_npy) = parseMatricesFromSeqs(pred.cpu().numpy(), targets.cpu().numpy(), tmp_dates, 
                                                        n_test_dates_target,
                                                        unique_tst_dates_target) 

        outputm_npy = np.transpose(outputm_npy)
        label_mat= np.transpose(labelm_npy)
        output_df = pd.DataFrame(data=outputm_npy, columns=['temp_pred'], index=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
        label_df = pd.DataFrame(data=label_mat, columns=['temp_actual'], index=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
        output_df.rename(columns={'Date': 'temp_pred'})
        label_df.rename(columns={'Date': 'temp_actual'})

        assert np.isfinite(np.array(output_df.values[:,1:],dtype=np.float32)).all(), "nan output"
        output_df['temp_actual'] = label_df['temp_actual']

        lake_output_path = '../../results/SWT_results/outputs_'+target_id+'.feather'
        # if not os.path.exists(lake_output_path):
        #     os.mkdir(lake_output_path)
        output_df = output_df[~output_df['index'].str.contains("1979")]
        output_df = output_df[~output_df['index'].str.contains("2021")]
        output_df.reset_index(inplace=True)
        
        output_df.to_feather(lake_output_path)
        #to store output
        # output_mats[i,:,:] = outputm_npy
        # loss_output = outputm_npy[~np.isnan(labelm_npy)]
        # loss_label = labelm_npy[~np.isnan(labelm_npy)]
        # loss_days = unique_tst_dates_target[~np.isnan(labelm_npy)]
        # # print(unique_tst_dates_target)
        # output_df = pd.DataFrame()
        # output_df['Date'] = loss_days
        # output_df['site_id'] = target_id
        # output_df['wtemp_predicted'] = loss_output
        # output_df['wtemp_actual'] =loss_label
        # output_df['fold'] = k


        # final_output_df = pd.concat([final_output_df, output_df],ignore_index=True)
        # mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())
        # print("globLSTM rmse(",loss_output.shape[0]," obs)=", mat_rmse)
        # if output_df.shape[0] != obs[obs['site_id']==target_id].shape[0]:
        #     print("missed obs?")
        #     pdb.set_trace()
# final_output_df.to_feather("../../results/err_est_outputs.feather")



print("DONE!")

#     load_path = '../../models/global_model_128hid_1layer_final_2feat_conus_intermediate'
#     n_hidden = torch.load(load_path)['state_dict']['lstm.weight_hh'].shape[0]
#     lstm_net = Model(input_size_dyn=n_features,input_size_stat=n_static_feats,hidden_size=n_hidden)
#     if use_gpu:
#         lstm_net = lstm_net.cuda(0)
#     pretrain_dict = torch.load(load_path)['state_dict']
#     model_dict = lstm_net.state_dict()
#     pretrain_dict = {key: v for key, v in pretrain_dict.items() if key in model_dict}
#     model_dict.update(pretrain_dict)
#     lstm_net.load_state_dict(pretrain_dict)

#     #things needed to predict test data
#     mse_criterion = nn.MSELoss()
#     testloader = torch.utils.data.DataLoader(tst_data_target, batch_size=tst_data_target.size()[0], shuffle=False, pin_memory=True)

#     lstm_net.eval()
#     with torch.no_grad():
#         avg_mse = 0
#         ct = 0
#         for m, data in enumerate(testloader, 0):
#             #now for mendota data
#             #this loop is dated, there is now only one item in testloader

#             #parse data into inputs and targets
#             inputs = data[:,:,:n_total_features].float()
#             targets = data[:,:,-1].float()
#             targets = targets[:, begin_loss_ind:]
#             tmp_dates = tst_dates[:, begin_loss_ind:]
#             depths = inputs[:,:,0]

#             if use_gpu:
#                 inputs = inputs.cuda()
#                 targets = targets.cuda()

#             #run model
#             # h_state = None
#             # lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
#             # pred, h_state = lstm_net(inputs, h_state)
#             pred, h_state, _ = lstm_net(inputs[:,:,n_static_feats:], inputs[:,0,:n_static_feats])
#             pred = pred.view(pred.size()[0],-1)
#             pred = pred[:, begin_loss_ind:]

#             #calculate error
#             targets = targets.cpu()
#             loss_indices = np.where(~np.isnan(targets))
#             if use_gpu:
#                 targets = targets.cuda()
#             inputs = inputs[:, begin_loss_ind:, :]
#             depths = depths[:, begin_loss_ind:]

#             mse = mse_criterion(pred[loss_indices], targets[loss_indices])
#             # print("test loss = ",mse)
#             avg_mse += mse

#             if mse > 0: #obsolete i think
#                 ct += 1
#             avg_mse = avg_mse / ct


#             #save model 
#             (outputm_npy, labelm_npy) = parseMatricesFromSeqs(pred.cpu().numpy(), targets.cpu().numpy(), tmp_dates, 
#                                                             n_test_dates_target,
#                                                             unique_tst_dates_target) 
#             #to store output
#             # output_mats[i,:,:] = outputm_npy
#             loss_output = outputm_npy[~np.isnan(labelm_npy)]
#             loss_label = labelm_npy[~np.isnan(labelm_npy)]
#             loss_days = unique_tst_dates_target[~np.isnan(labelm_npy)]
#             # outputs.append(loss_output)
#             # labels.append(loss_label)
#             # dates.append(loss_days)
#             site_id_repeat = np.empty((loss_label.shape[0]),dtype=np.object)
#             site_id_repeat[:] = ''
#             site_id_repeat[:] = target_id
#             sites.append(site_id_repeat)

#             mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())
#             print("globLSTM rmse(",loss_output.shape[0]," obs)=", mat_rmse)
#             # err_per_source[i,targ_ct] = mat_rmse

#             # glm_rmse = float(metadata.loc["nhdhr_"+target_id].glm_uncal_rmse_full)

#             # csv.append(",".join(["nhdhr_"+target_id,str(mat_rmse)]))
#             rmse_per_lake[targ_ct] = mat_rmse

#             outputm_npy = np.transpose(outputm_npy)
#             label_mat= np.transpose(labelm_npy)
#             output_df = pd.DataFrame(data=outputm_npy, columns=['temp_pred'], index=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
#             label_df = pd.DataFrame(data=label_mat, columns=['temp_actual'], index=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
#             output_df.rename(columns={'Date': 'temp_pred'})
#             label_df.rename(columns={'Date': 'temp_actual'})

#             assert np.isfinite(np.array(output_df.values[:,1:],dtype=np.float32)).all(), "nan output"
#             output_df['temp_actual'] = label_df['temp_actual']

#             lake_output_path = '../../results/outputs_'+target_id+'.feather'
#             # if not os.path.exists(lake_output_path):
#             #     os.mkdir(lake_output_path)
#             pdb.set_trace()
#             output_df.to_feather(lake_output_path)

#     # #save model 
#     # total_output_npy = np.average(output_mats, axis=0)

#     # if output_to_file:
#     #     outputm_npy = np.transpose(total_output_npy)
#     #     label_mat= np.transpose(label_mats)
#     #     output_df = pd.DataFrame(data=outputm_npy, columns=[str(float(x/2)) for x in range(outputm_npy.shape[1])], index=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
#     #     label_df = pd.DataFrame(data=label_mat, columns=[str(float(x/2)) for x in range(label_mat.shape[1])], index=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
#     #     output_df.rename(columns={'index': 'depth'})
#     #     label_df.rename(columns={'index': 'depth'})

#     #     assert np.isfinite(np.array(output_df.values[:,1:],dtype=np.float32)).all(), "nan output"
#     #     lake_output_path = output_path+target_id
#     #     if not os.path.exists(lake_output_path):
#     #         os.mkdir(lake_output_path)
#     #     output_df.to_feather(lake_output_path+"/PGMTL_outputs.feather")
        
#     # loss_output = total_output_npy[~np.isnan(label_mats)]
#     # loss_label = label_mats[~np.isnan(label_mats)]
#     # mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())

#     # print("Total rmse=", mat_rmse)
#     # spcorr = srcorr_per_lake[targ_ct]
#     # rmse_per_lake[targ_ct] = mat_rmse




# # with open(save_file_path,'w') as file:
# #     for line in csv:
# #         file.write(line)
# #         file.write('\n')




# # print("median srcorr: ",np.median(srcorr_per_lake))
# # print("median meta test RMSE(med): ",np.median(med_meta_rmse_per_lake))

# print("median test RMSE: ",np.median(rmse_per_lake))
# print("q1 test RMSE: ",np.quantile(rmse_per_lake,.25))
# print("q3 test RMSE: ",np.quantile(rmse_per_lake,.75))
# labels = np.concatenate(labels).reshape(-1)
# outputs = np.concatenate(outputs).reshape(-1)
# dates = np.concatenate(dates).reshape(-1)
# sites = np.concatenate(sites).reshape(-1)

# pdb.set_trace()
