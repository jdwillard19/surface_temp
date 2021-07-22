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
from pytorch_data_operations import buildLakeDataForRNN_repr_trn, buildLakeDataForRNN_repr_tst, parseMatricesFromSeqs


#get site Iids
site_ids = np.load("../../metadata/lakeset.npy",allow_pickle=True)

n_rand = int(sys.argv[1])
trn_path = "./ctlstm_"+str(n_rand)+"rand_trn_data.npy"
trn_date_path = "./ctlstm_"+str(n_rand)+"rand_trn_dates.npy"


#load train data
if not os.path.exists(trn_path):
    (trn_data, trn_dates) = buildLakeDataForRNN_repr_trn(site_ids,randomFeat=n_rand) 
    np.save(trn_path,trn_data)
    np.save(trn_date_path,trn_dates)
else:
    trn_data = torch.from_numpy(np.load(trn_path))
    trn_dates = np.load(trn_date_path,allow_pickle=True)





#params
use_gpu = True
verbose = True
grad_clip = 1.0 #how much to clip the gradient 2-norm in training
dropout = 0.
num_layers = 1
n_hidden = 256
# lambda1 = 1e-
lambda1 = 0.000

# n_eps = 10000
n_eps = 1000
targ_ep = 30
targ_rmse = 2.03

#
n_features = trn_data.shape[2]-1
n_static_feats = 0

print("train_data size: ",trn_data.size())
print(len(site_ids), " lakes of data")
# trn_data = tst_data
# batch_size = int(math.floor(trn_data.size()[0])/150)
# batch_size = int(math.floor(trn_data.size()[0])/20)
batch_size = 1500 #maybe have to adjust
if n_rand == 512:
    batch_size = 750
n_runs = 5

# batch_size = trn_data.size()[0] #DEBUG VALUE

train = [True,True,True,True,True]
# train = [False,False,False,False,False]


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




#format training data for loading
train_data = TemperatureTrainDataset(trn_data)


#format total y-hat data for loading
# total_data = TotalModelOutputDataset(all_data, all_phys_data, all_dates)
n_batches = math.floor(trn_data.size()[0] / batch_size)

#batch samplers used to draw samples in dataloaders
batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)


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
    """Implementation of the standard LSTM.
    TODO: Include ref and LaTeX equations
    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # create tensors of learnable parameters
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        nn.init.orthogonal_(self.weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data = weight_hh_data
        nn.init.constant_(self.bias.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[summary]
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor, containing a batch of input sequences. Format must match the specified format,
            defined by the batch_first agrument.
        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor]
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()

        h_0 = x.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh) + torch.mm(x[t], self.weight_ih))
            f, i, o, g = gates.chunk(4, 1)

            c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
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

final_output_df = pd.DataFrame()
for r in range(n_runs):
    # lstm_net = myLSTM_Net(n_total_feats, n_hidden, batch_size)
    lstm_net = Model(input_size_dyn=n_features-n_static_feats,input_size_stat=n_static_feats,hidden_size=n_hidden,no_static=True)
    save_path = '../../models/CTLSTM_'+str(n_rand)+'rand_run'+str(r)

    if not train[r]:
        continue
        # load_path = save_path
        # n_hidden = torch.load(load_path)['state_dict']['lstm.weight_hh'].shape[0]
        # lstm_net = Model(input_size_dyn=n_features,input_size_stat=n_static_feats,hidden_size=n_hidden,no_static=True)
        # if use_gpu:
        #     lstm_net = lstm_net.cuda(0)
        # pretrain_dict = torch.load(load_path)['state_dict']
        # model_dict = lstm_net.state_dict()
        # pretrain_dict = {key: v for key, v in pretrain_dict.items() if key in model_dict}
        # model_dict.update(pretrain_dict)
        # lstm_net.load_state_dict(pretrain_dict)
        # mse_criterion = nn.MSELoss()

    else:
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
        min_train_rmse = 999
        min_train_ep = -1
        done = False
        for epoch in range(n_eps):
            if done:
                break
            # if verbose and epoch % 10 == 0:
            if verbose:
                print("r",r,", ","train epoch: ", epoch)

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
                # tmp_dates = tst_dates
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


                loss_outputs = outputs[:,:]
                loss_targets = targets[:,:].cpu()


                #get indices to calculate loss
                loss_indices = np.array(np.isfinite(loss_targets.cpu()), dtype='bool_')

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
            # if verbose and epoch %100 is 0:


            if verbose:
                print("train rmse loss=", avg_loss)
            
            if avg_loss < min_train_rmse:
                min_train_rmse = avg_loss
                print("model saved")
                saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)


            # if epoch % 10 is 0:
            if avg_loss < targ_rmse and epoch > targ_ep:
                print("training complete")
                print("model saved")
                saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
                break


#test model

final_output_df = pd.DataFrame()
models = []
#load all models for testing
for r in range(n_runs):
    # lstm_net = myLSTM_Net(n_total_feats, n_hidden, batch_size)
    # lstm_net = Model(input_size_dyn=n_features,input_size_stat=n_static_feats,hidden_size=n_hidden,no_static=True)
    load_path = '../../models/CTLSTM_'+str(n_rand)+'rand_run'+str(r)
    n_hidden = torch.load(load_path)['state_dict']['lstm.weight_hh'].shape[0]
    lstm_net = Model(input_size_dyn=n_features-n_static_feats,input_size_stat=n_static_feats,\
                     hidden_size=n_hidden,no_static=True)
    if use_gpu:
        lstm_net = lstm_net.cuda(0)
    pretrain_dict = torch.load(load_path)['state_dict']
    model_dict = lstm_net.state_dict()
    pretrain_dict = {key: v for key, v in pretrain_dict.items() if key in model_dict}
    model_dict.update(pretrain_dict)
    lstm_net.load_state_dict(pretrain_dict)
    mse_criterion = nn.MSELoss()
    models.append(lstm_net)

#test model
rmse_per_lake = np.empty((len(site_ids)))
rmse_per_lake[:] = np.nan
for targ_ct, target_id in enumerate(site_ids): #for each target lake
    print("testing lake ",targ_ct,"/",len(site_ids))
    site_df = pd.DataFrame()
    lake_id = target_id


    #target agnostic model and data params
    (tst_data, tst_dates, all_dates) = buildLakeDataForRNN_repr_tst([target_id],randomFeat=n_rand) 


    #useful values, LSTM params
    batch_size = tst_data.size()[0]
    n_test_dates = all_dates.shape[0]

    testloader = torch.utils.data.DataLoader(tst_data, batch_size=tst_data.size()[0], shuffle=False, pin_memory=True)

    for r in range(n_runs):
        with torch.no_grad():
            avg_mse = 0
            ct = 0
            for m, data in enumerate(testloader, 0):
                #now for mendota data
                #this loop is dated, there is now only one item in testloader

                #parse data into inputs and targets
                inputs = data[:,:,:-1].float()
                targets = data[:,:,-1].float()
                tmp_dates = tst_dates[:, :]

                if use_gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                #run model
                h_state = None
                # lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
                # outputs, h_state, c_state = lstm_net(inputs[:,:,:n_features], inputs[:,0,n_features:])
                pred, h_state, _ = models[r](inputs[:,:,n_static_feats:], inputs[:,0,:n_static_feats])
                pred = pred.view(pred.size()[0],-1)
                pred = pred[:, :]

                #calculate error
                targets = targets.cpu()
                loss_indices = np.where(np.isfinite(targets))
                if use_gpu:
                    targets = targets.cuda()
                inputs = inputs[:, :, :]
                mse = mse_criterion(pred[loss_indices], targets[loss_indices])
                # print("test loss = ",mse)
                avg_mse += mse
                ct += 1
                # if mse > 0: #obsolete i think
                #     ct += 1
            avg_mse = avg_mse / ct

            (outputm_npy, labelm_npy) = parseMatricesFromSeqs(pred.cpu().numpy(), targets.cpu().numpy(), tmp_dates, 
                                                            n_test_dates,
                                                            all_dates) 


            #to store output
            # output_mats[i,:,:] = outputm_npy
            loss_output = outputm_npy[~np.isnan(labelm_npy)]
            loss_label = labelm_npy[~np.isnan(labelm_npy)]
            loss_days = all_dates[~np.isnan(labelm_npy)]

            if r == 0:
                site_df['date'] = loss_days
                site_df['actual'] = loss_label
                site_df['pred'+str(r)] = loss_output
            else:
                site_df['pred'+str(r)] = loss_output

    site_df['avg_pred'] = site_df.iloc[:,2:(2+n_runs)].mean(axis=1)

    mat_rmse = np.sqrt(((site_df['avg_pred'] - site_df['actual']) ** 2).mean())
    rmse_per_lake[targ_ct] = mat_rmse
    print("globLSTM rmse(",loss_output.shape[0]," obs)=", mat_rmse)
# final_output_df.to_feather("../../results/err_est_outputs_225hid_EALSTM_fold"+str(k)+".feather")
final_output_df = pd.DataFrame()
final_output_df['site_id'] = site_ids
final_output_df['rmse'] = rmse_per_lake
final_output_df.reset_index(inplace=True)
final_output_df.to_csv("../../results/CTLSTM_"+str(n_rand)+'rand.csv')