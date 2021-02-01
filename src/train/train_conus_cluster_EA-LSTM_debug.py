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
from pytorch_data_operations import buildLakeDataForRNNPretrain
from pytorch_model_operations import saveModel
import pytorch_data_operations
import datetime
import pdb
from torch.utils.data import DataLoader
from pytorch_data_operations import buildLakeDataForRNN_multilakemodel_conus, parseMatricesFromSeqs



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
debug_train = False
debug_end = False
verbose = True
save = True
test = False



#####################3
#params
###########################33
first_save_epoch = 0
patience = 100

#ow
seq_length = 350 #how long of sequences to use in model
begin_loss_ind = 0#index in sequence where we begin to calculate error or predict
n_features = 5  #number of physical drivers
n_static_feats = 3
n_total_feats =n_static_feats+n_features
win_shift = 175 #how much to slide the window on training set each time
save = True 
grad_clip = 1.0 #how much to clip the gradient 2-norm in training
dropout = 0.
num_layers = 1
n_hidden = 128
# lambda1 = 1e-
lambda1 = 0

n_eps = 10000
# n_ep/rmse = (1013/1.52)(957/1.51?
targ_ep = 0
targ_rmse = 1.46
ep_list16 = [] #list of epochs at which models were saved for * hidden units
ep_list32 = [] 
ep_list64 = [] 
ep_list128 = [] 

cluster = sys.argv[1]
metadata = pd.read_csv("../../metadata/conus_source_metadata.csv")
lakenames = metadata[metadata['cluster']==int(cluster)]['site_id'].values
# lakenames = np.load("../../data/static/lists/source_lakes_conus.npy",allow_pickle=True)


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

(trn_data, _) = buildLakeDataForRNN_multilakemodel_conus(lakenames,\
                                                seq_length, n_total_feats,\
                                                win_shift = win_shift, begin_loss_ind = begin_loss_ind,\
                                                ) 
# np.save("conus_trn_data_wStatic.npy",trn_data)
# np.save("_tst_data_wStatic.npy",tst_data)
# sys.exit()
# trn_data = torch.from_numpy(np.load("conus_trn_data_wStatic.npy"))
# tst_data = torch.from_numpy(np.load("global_tst_data_wStatic.npy"))
# tst_data = tst_data[:,:,[0,1,2,4,7,-1]]

# trn_data = tst_data
# n_features = 4
# n_static_feats = 1
# n_total_feats = n_features + n_static_feats
print("train_data size: ",trn_data.size())
print(len(lakenames), " lakes of data")
# trn_data = tst_data
# batch_size = trn_data.size()[0]
batch_size = int(math.floor(trn_data.size()[0])/20)



#Dataset classes
class TemperatureTrainDataset(Dataset):
#training dataset class, allows Dataloader to load both input/target
    def __init__(self, trn_data):
        self.len = trn_data.shape[0]
        self.x_data = trn_data[:,:,:-1].float()
        self.y_data = trn_data[:,:,-1].float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# class TotalModelOutputDataset(Dataset):
# #dataset for unsupervised input(in this case all the data)
#     def __init__(self, all_data, all_phys_data,all_dates):
#         #data of all model output, and corresponding unstandardized physical quantities
#         #needed to calculate physical loss
#         self.len = all_data.shape[0]
#         self.data = all_data[:,:,:-1].float()
#         self.label = all_data[:,:,-1].float() #DO NOT USE IN MODEL
#         self.phys = all_phys_data.float()
#         helper = np.vectorize(lambda x: date.toordinal(pd.Timestamp(x).to_pydatetime()))
#         dates = helper(all_dates)
#         self.dates = dates

#     def __getitem__(self, index):
#         return self.data[index], self.phys[index], self.dates[index], self.label[index]

#     def __len__(self):
#         return self.len




#format training data for loading
train_data = TemperatureTrainDataset(trn_data)


#format total y-hat data for loading
# total_data = TotalModelOutputDataset(all_data, all_phys_data, all_dates)
n_batches = math.floor(trn_data.size()[0] / batch_size)

#batch samplers used to draw samples in dataloaders
batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)


#load val/test data into enumerator based on batch size
# testloader = torch.utils.data.DataLoader(tst_data, batch_size=tst_data.size()[0], shuffle=False, pin_memory=True)


#define EA-LSTM class
"""
This file is part of the accompanying code to the manuscript:
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




#define loss and optimizer
mse_criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_net.parameters(), lr=.005)#, weight_decay=0.01)

#training loop

min_mse = 99999
min_mse_tsterr = None
ep_min_mse = -1
ep_since_min = 0
best_pred_mat = np.empty(())
manualSeed = [random.randint(1, 99999999) for i in range(n_eps)]

#stop training if true
done = False
min_train_rmse = 999
min_train_ep = -1
for epoch in range(n_eps):
    # if verbose and epoch % 10 == 0:
    if verbose:
        print("train epoch: ", epoch)

    if done:
        break

    running_loss = 0.0

    #reload loader for shuffle
    #batch samplers used to draw samples in dataloaders
    batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)

    trainloader = DataLoader(train_data, batch_sampler=batch_sampler, pin_memory=True)


    #zero the parameter gradients
    optimizer.zero_grad()
    lstm_net.train(True)
    avg_loss = 0
    batches_done = 0
    ct = 0
    for m, data in enumerate(trainloader, 0):
        #now for mendota data
        #this loop is dated, there is now only one item in testloader

        #parse data into inputs and targets
        inputs = data[0].float()
        targets = data[1].float()
        targets = targets[:, begin_loss_ind:]
        # tmp_dates = tst_dates_target[:, begin_loss_ind:]
        # depths = inputs[:,:,0]


        #cuda commands
        if(use_gpu):
            inputs = inputs.cuda()
            targets = targets.cuda()

        #forward  prop
        # lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
        # lstm_net.reset_parameters()
        # h_state = None
        outputs, h_state, _ = lstm_net(inputs[:,:,n_static_feats:], inputs[:,0,:n_static_feats])
        outputs = outputs.view(outputs.size()[0],-1)

        #calculate losses
        reg1_loss = 0
        if lambda1 > 0:
            reg1_loss = calculate_l1_loss(lstm_net)


        loss_outputs = outputs[:,begin_loss_ind:]
        loss_targets = targets[:,begin_loss_ind:].cpu()


        #get indices to calculate loss
        loss_indices = torch.from_numpy(np.array(np.isfinite(loss_targets.cpu()), dtype='bool_'))

        if use_gpu:
            loss_outputs = loss_outputs.cuda()
            loss_targets = loss_targets.cuda()
        loss = mse_criterion(loss_outputs[loss_indices], loss_targets[loss_indices]) + lambda1*reg1_loss 
        #backward

        loss.backward(retain_graph=False)
        if grad_clip > 0:
            clip_grad_norm_(lstm_net.parameters(), grad_clip, norm_type=2)

        #optimize
        optimizer.step()

        #zero the parameter gradients
        optimizer.zero_grad()
        avg_loss += loss
        batches_done += 1

    #check for convergence
    avg_loss = avg_loss / batches_done
    train_avg_loss = avg_loss
    # if verbose and epoch %100 is 0:
    if verbose:
        print("rmse loss=", avg_loss)
    if avg_loss < targ_rmse and epoch > targ_ep:
        break

    # if avg_loss < min_mse:
    #     min_mse = avg_loss
    #     ep_min_mse = epoch
    #     ep_since_min = 0

    # else:
    #     ep_since_min += 1

    # if ep_since_min == patience:
    #     print("patience met")
    #     done = True
    #     break

    # if test:
    #     with torch.no_grad():
    #         avg_mse = 0
    #         ct = 0
    #         for m, data in enumerate(testloader, 0):
    #             #now for mendota data
    #             #this loop is dated, there is now only one item in testloader

    #             #parse data into inputs and targets
    #             inputs = data[:,:,:n_total_feats].float()
    #             targets = data[:,:,-1].float()
    #             targets = targets[:, begin_loss_ind:]

    #             if use_gpu:
    #                 inputs = inputs.cuda()
    #                 targets = targets.cuda()

    #             #run model
    #             h_state = None
    #             # lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
    #             # outputs, h_state, c_state = lstm_net(inputs[:,:,:n_features], inputs[:,0,n_features:])
    #             pred, h_state, _ = lstm_net(inputs[:,:,:n_features], inputs[:,0,n_features:])
    #             pred = pred.view(pred.size()[0],-1)
    #             pred = pred[:, begin_loss_ind:]

    #             #calculate error
    #             targets = targets.cpu()
    #             loss_indices = np.where(np.isfinite(targets))
    #             if use_gpu:
    #                 targets = targets.cuda()
    #             inputs = inputs[:, begin_loss_ind:, :]
    #             mse = mse_criterion(pred[loss_indices], targets[loss_indices])
    #             # print("test loss = ",mse)
    #             avg_mse += mse

    #             # if mse > 0: #obsolete i think
    #             #     ct += 1
    #             # avg_mse = avg_mse / ct


    #             # #save model 
    #             # (outputm_npy, labelm_npy) = parseMatricesFromSeqs(pred.cpu().numpy(), targets.cpu().numpy(), tmp_dates, 
    #             #                                                 n_test_dates_target,
    #             #                                                 unique_tst_dates_target) 
    #             # #to store output
    #             # output_mats[i,:] = outputm_npy
    #             # if i == 0:
    #             #     #store label
    #             #     label_mats = labelm_npy
    #             # loss_output = outputm_npy[~np.isnan(labelm_npy)]
    #             # loss_label = labelm_npy[~np.isnan(labelm_npy)]

    #             # avg_mse = np.sqrt(((loss_output - loss_label) ** 2).mean())

    #             if avg_mse < min_mse:
    #                 # save_path = "../../models/global_model_"+str(n_hidden)+"hid_"+str(num_layers)+"layer_"+str(dropout)+"drop"
    #                 # saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
    #                 min_train_ep = epoch
    #                 min_train_rmse = train_avg_loss
    #                 min_mse = avg_mse
    #                 ep_since_min = 0
    #             else:
    #                 ep_since_min += 1
    #                 if ep_since_min == patience:
    #                     print("patience met")
    #                     print("min train ep/rmse: ",min_train_ep,"\n",min_train_rmse)
    #                     sys.exit()

    #             print("Test RMSE: ", avg_mse, "(min=",min_mse,")---ep since ",ep_since_min)

        # if epoch % 100 == 0 and epoch != 0:

    #     save_path = "../../models/"+lakename+"/basicLSTM_source_model_"+str(n_hidden)+"hid_"+str(epoch)+"ep"

    #     saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
    #     if n_hidden == n_hidden_list[0]:
    #         ep_list16.append(epoch)
    #     elif n_hidden == n_hidden_list[1]:
    #         ep_list32.append(epoch)
    #     elif n_hidden is n_hidden_list[2]:
    #         ep_list64.append(epoch)
    #     elif n_hidden is n_hidden_list[3]:
    #         ep_list128.append(epoch)
save_path = "../../models/conus_cluster_model_"+str(n_hidden)+"hid_"+str(num_layers)+"layer_final_"+str(cluster)

saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)

print("saved at ",save_path)



# print("|\n|\nTraining Candidate Models Complete\n|\n|")
# ##################################################################
# # transfer all models to all other source lakes to find best one
# ########################################################################

# #load all other source lakes
# glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
# train_lakes = np.array([re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)])

# other_source_ids = train_lakes[~np.isin(train_lakes,site_id)] #remove site id

# n_other_sources = len(other_source_ids)


# top_ids = [site_id]
# err_per_16hid_ep = np.empty((n_other_sources,len(ep_list16)))
# # err_per_hid_ep20 = np.empty((len(ep_list20)))
# err_per_32hid_ep = np.empty((n_other_sources,len(ep_list32)))
# err_per_64hid_ep = np.empty((n_other_sources,len(ep_list64)))
# err_per_128hid_ep = np.empty((n_other_sources,len(ep_list128)))

# for hid_ct, n_hidden in enumerate(n_hidden_list):
#     ep_list = []
#     if n_hidden == n_hidden_list[0]:
#         ep_list = ep_list16
#     elif n_hidden == n_hidden_list[1]:
#         ep_list = ep_list32
#     elif n_hidden == n_hidden_list[2]:
#         ep_list = ep_list64
#     elif n_hidden == n_hidden_list[3]:
#         ep_list = ep_list128


#     for ep_ct, eps in enumerate(ep_list):
#         for targ_ct,target_id in enumerate(other_source_ids):
#             print("TARGET: ", target_id)
#             data_dir_target = "../../data/processed/"+target_id+"/" 
#             #target agnostic model and data params
#             use_gpu = True
#             n_features = 7
#             seq_length = 350
#             win_shift = 175
#             begin_loss_ind = 0
#             (_, _, tst_data_target, tst_dates_target, unique_tst_dates_target, all_data_target, \
#              all_phys_data_target, all_dates_target)\
#             = buildLakeDataForRNN_manylakes_finetune2(target_id, data_dir_target, seq_length, n_features,
#                                                win_shift = win_shift, begin_loss_ind = begin_loss_ind, 
#                                                outputFullTestMatrix=True, allTestSeq=True)
            

#             #useful values, LSTM params
#             batch_size = all_data_target.size()[0]
#             n_test_dates_target = unique_tst_dates_target.shape[0]


#             #define LSTM model
#             class LSTM(nn.Module):
#                 def __init__(self, input_size, hidden_size, batch_size):
#                     super(LSTM, self).__init__()
#                     self.input_size = input_size
#                     self.hidden_size = hidden_size
#                     self.batch_size = batch_size
#                     self.lstm = nn.LSTM(input_size = n_features, hidden_size=hidden_size, batch_first=True, num_layers=num_layers,dropout=dropout) 
#                     self.out = nn.Linear(hidden_size, 1)
#                     self.hidden = self.init_hidden()

#                 def init_hidden(self, batch_size=0):
#                     # initialize both hidden layers
#                     if batch_size == 0:
#                         batch_size = self.batch_size
#                     ret = (xavier_normal_(torch.empty(num_layers, batch_size, self.hidden_size)),
#                             xavier_normal_(torch.empty(num_layers, batch_size, self.hidden_size)))
#                     if use_gpu:
#                         item0 = ret[0].cuda(non_blocking=True)
#                         item1 = ret[1].cuda(non_blocking=True)
#                         ret = (item0,item1)
#                     return ret
                
#                 def forward(self, x, hidden): #forward network propagation 
#                     self.lstm.flatten_parameters()
#                     x = x.float()
#                     x, hidden = self.lstm(x, self.hidden)
#                     self.hidden = hidden
#                     x = self.out(x)
#                     return x, hidden



#             #output matrix
#             n_lakes = len(top_ids)
#             output_mats = np.empty((n_lakes, n_test_dates_target))
#             ind_rmses = np.empty((n_lakes))
#             ind_rmses[:] = np.nan
#             label_mats = np.empty((n_test_dates_target)) 
#             output_mats[:] = np.nan
#             label_mats[:] = np.nan


#             for i, source_id in enumerate(top_ids): 
#                 #for each top id
#                 # source_id = re.search('nhdhr_(.*)', source_id).group(1)
#                 #load source model
#                 load_path = "../../models/"+source_id+"/basicLSTM_source_model_"+str(n_hidden)+"hid_"+str(eps)+"ep"
#                 lstm_net = LSTM(n_features, n_hidden, batch_size)
#                 if use_gpu:
#                     lstm_net = lstm_net.cuda(0)
#                 pretrain_dict = torch.load(load_path)['state_dict']
#                 model_dict = lstm_net.state_dict()
#                 pretrain_dict = {key: v for key, v in pretrain_dict.items() if key in model_dict}
#                 model_dict.update(pretrain_dict)
#                 lstm_net.load_state_dict(pretrain_dict)

#                 #things needed to predict test data
#                 mse_criterion = nn.MSELoss()
#                 testloader = torch.utils.data.DataLoader(tst_data_target, batch_size=tst_data_target.size()[0], shuffle=False, pin_memory=True)

#                 lstm_net.eval()
#                 with torch.no_grad():
#                     avg_mse = 0
#                     ct = 0
#                     for m, data in enumerate(testloader, 0):
#                         #now for mendota data
#                         #this loop is dated, there is now only one item in testloader

#                         #parse data into inputs and targets
#                         inputs = data[:,:,:n_features].float()
#                         targets = data[:,:,-1].float()
#                         targets = targets[:, begin_loss_ind:]
#                         tmp_dates = tst_dates_target[:, begin_loss_ind:]
#                         depths = inputs[:,:,0]

#                         if use_gpu:
#                             inputs = inputs.cuda()
#                             targets = targets.cuda()

#                         #run model
#                         h_state = None
#                         lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
#                         pred, h_state = lstm_net(inputs, h_state)
#                         pred = pred.view(pred.size()[0],-1)
#                         pred = pred[:, begin_loss_ind:]

#                         #calculate error
#                         targets = targets.cpu()
#                         loss_indices = np.where(~np.isnan(targets))
#                         if use_gpu:
#                             targets = targets.cuda()
#                         inputs = inputs[:, begin_loss_ind:, :]
#                         depths = depths[:, begin_loss_ind:]
#                         mse = mse_criterion(pred[loss_indices], targets[loss_indices])
#                         # print("test loss = ",mse)
#                         avg_mse += mse

#                         if mse > 0: #obsolete i think
#                             ct += 1
#                         avg_mse = avg_mse / ct


#                         #save model 
#                         (outputm_npy, labelm_npy) = parseMatricesFromSeqs(pred.cpu().numpy(), targets.cpu().numpy(), tmp_dates, 
#                                                                         n_test_dates_target,
#                                                                         unique_tst_dates_target) 
#                         #to store output
#                         output_mats[i,:] = outputm_npy
#                         if i == 0:
#                             #store label
#                             label_mats = labelm_npy
#                         loss_output = outputm_npy[~np.isnan(labelm_npy)]
#                         loss_label = labelm_npy[~np.isnan(labelm_npy)]

#                         mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())
#                         # print(source_id+" rmse=", mat_rmse)

#                         # glm_rmse = float(metadata.loc["nhdhr_"+target_id].glm_uncal_rmse_full)

#                         # mat_csv.append(",".join(["nhdhr_"+target_id,"nhdhr_"+ source_id,str(meta_rmse_per_lake[targ_ct]),str(srcorr_per_lake[targ_ct]), str(glm_rmse),str(mat_rmse)] + [str(x) for x in lake_df.iloc[i][feats].values]))


#             #save model 
#             total_output_npy = np.average(output_mats, axis=0)

#             loss_output = total_output_npy[~np.isnan(label_mats)]
#             loss_label = label_mats[~np.isnan(label_mats)]
#             mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())

#             print("source ",site_id, "-> target ", target_id,": Total rmse=", mat_rmse)


#             if n_hidden == n_hidden_list[0]:
#                 err_per_16hid_ep[targ_ct, ep_ct] = mat_rmse
#             elif n_hidden == n_hidden_list[1]:
#                 err_per_32hid_ep[targ_ct, ep_ct] = mat_rmse
#             elif n_hidden == n_hidden_list[2]:
#                 err_per_64hid_ep[targ_ct, ep_ct] = mat_rmse
#             elif n_hidden == n_hidden_list[3]:
#                 err_per_128hid_ep[targ_ct, ep_ct] = mat_rmse



# best_hid = None
# best_ep = None

# min_ep_ind_16hid = np.argmin(err_per_16hid_ep)
# best_ep_16hid = (min_ep_ind_16hid+1)*100

# min_ep_ind_32hid = np.argmin(err_per_32hid_ep)
# best_ep_32hid = (min_ep_ind_32hid+1)*100

# min_ep_ind_64hid = np.argmin(err_per_64hid_ep)
# best_ep_64hid = (min_ep_ind_64hid+1)*100

# min_ep_ind_128hid = np.argmin(err_per_128hid_ep)
# best_ep_128hid = (min_ep_ind_128hid+1)*100


# print("BEST_MODEL_PATH_16HID:../../models/"+str(lakename)+"/basicLSTM_source_model_16hid_"+str(best_ep_16hid)+"ep\nRMSE="+str(err_per_16hid_ep.min()))
# print("BEST_MODEL_PATH_32HID:../../models/"+str(lakename)+"/basicLSTM_source_model_32hid_"+str(best_ep_32hid)+"ep\nRMSE="+str(err_per_32hid_ep.min()))
# print("BEST_MODEL_PATH_64HID:../../models/"+str(lakename)+"/basicLSTM_source_model_64hid_"+str(best_ep_64hid)+"ep\nRMSE="+str(err_per_64hid_ep.min()))
# print("BEST_MODEL_PATH_128HID:../../models/"+str(lakename)+"/basicLSTM_source_model_128hid_"+str(best_ep_128hid)+"ep\nRMSE="+str(err_per_128hid_ep.min()))
# # with open(save_file_path,'w') as file:
# #     for line in mat_csv:
# #         file.write(line)
# #         file.write('\n')


