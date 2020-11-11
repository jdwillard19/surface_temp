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
import os
sys.path.append('../../data')
sys.path.append('../data')
sys.path.append('../../models')
sys.path.append('../models')
from pytorch_data_operations import buildLakeDataForRNNPretrain, calculate_ec_loss_manylakes, calculate_dc_loss
from pytorch_model_operations import saveModel
import pytorch_data_operations
import datetime
import pdb
from torch.utils.data import DataLoader
from pytorch_data_operations import buildLakeDataForRNN_manylakes_finetune2, parseMatricesFromSeqs



#script start
currentDT = datetime.datetime.now()
print(str(currentDT))


####################################################3
# (Nov 2020 - Jared) source model script, takes lakename as required command line argument
###################################################33

#enable/disable cuda 
use_gpu = True 
torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=10)

#cmd args
site_id = sys.argv[1]


### debug tools
debug_train = False
debug_end = False
verbose = True
pretrain = True
save = True
save_pretrain = True

#RMSE threshold for pretraining
num_layers = 1



#####################3
#params
###########################33
first_save_epoch = 0
patience = 100

n_hidden_list = [20,50] #fixed
train_epochs = 10000
pretrain_epochs = 10000

unsup_loss_cutoff = 40
dc_unsup_loss_cutoff = 1e-3
dc_unsup_loss_cutoff2 = 1e-2
#ow
seq_length = 350 #how long of sequences to use in model
begin_loss_ind = 0#index in sequence where we begin to calculate error or predict
n_features = 7  #number of physical drivers
win_shift = 50 #how much to slide the window on training set each time
save = True 


lakename = site_id
print("lake: "+lakename)
data_dir = "../../data/processed/"+lakename+"/"

###############################
# data preprocess
##################################
#create train and test sets

for n_hidden in n_hidden_list:
    #####################################################################################
    ####################################################3
    # fine tune
    ###################################################33
    ##########################################################################################33

    #####################
    #params
    ###########################
    first_save_epoch = 0
    n_eps = 2000
    patience = 1000
    epoch_since_best = 0
    lambda1 = 0.0001
    data_dir = "../../data/processed/"+lakename+"/"
    yhat_batch_size = 1

    ###############################
    # data preprocess
    ##################################
    #create train and test sets
    (trn_data, trn_dates, tst_data, tst_dates, unique_tst_dates, all_data, all_phys_data, all_dates) = buildLakeDataForRNN_manylakes_finetune2(lakename, data_dir, seq_length, n_features,
                                   win_shift = win_shift, begin_loss_ind = begin_loss_ind, 
                                   outputFullTestMatrix=True, allTestSeq=True) 

    trn_data = tst_data
    batch_size = trn_data.size()[0]



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

    class TotalModelOutputDataset(Dataset):
    #dataset for unsupervised input(in this case all the data)
        def __init__(self, all_data, all_phys_data,all_dates):
            #data of all model output, and corresponding unstandardized physical quantities
            #needed to calculate physical loss
            self.len = all_data.shape[0]
            self.data = all_data[:,:,:-1].float()
            self.label = all_data[:,:,-1].float() #DO NOT USE IN MODEL
            self.phys = all_phys_data.float()
            helper = np.vectorize(lambda x: date.toordinal(pd.Timestamp(x).to_pydatetime()))
            dates = helper(all_dates)
            self.dates = dates

    def __getitem__(self, index):
        return self.data[index], self.phys[index], self.dates[index], self.label[index]

    def __len__(self):
        return self.len




    #format training data for loading
    train_data = TemperatureTrainDataset(trn_data)


    #format total y-hat data for loading
    total_data = TotalModelOutputDataset(all_data, all_phys_data, all_dates)
    n_batches = math.floor(trn_data.size()[0] / batch_size)

    #batch samplers used to draw samples in dataloaders
    batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)


    #load val/test data into enumerator based on batch size
    testloader = torch.utils.data.DataLoader(tst_data, batch_size=tst_data.size()[0], shuffle=False, pin_memory=True)


    #define LSTM model class
    class myLSTM_Net(nn.Module):
        def __init__(self, input_size, hidden_size, batch_size):
            super(myLSTM_Net, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.batch_size = batch_size
            self.lstm = nn.LSTM(input_size = n_features, hidden_size=hidden_size, batch_first=True,num_layers=num_layers) #batch_first=True?
            self.out = nn.Linear(hidden_size, 1) #1?
            self.hidden = self.init_hidden()
            self.w_upper_to_lower = []
            self.w_lower_to_upper = []

        def init_hidden(self, batch_size=0):
            # initialize both hidden layers
            if batch_size == 0:
                batch_size = self.batch_size
            ret = (xavier_normal_(torch.empty(1, batch_size, self.hidden_size)),
                    xavier_normal_(torch.empty(1, batch_size, self.hidden_size)))
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


    lstm_net = myLSTM_Net(n_features, n_hidden, batch_size)

    #tell model to use GPU if needed
    if use_gpu:
        lstm_net = lstm_net.cuda()




    #define loss and optimizer
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_net.parameters(), lr=.001)#, weight_decay=0.01)

    #training loop

    min_mse = 99999
    min_mse_tsterr = None
    ep_min_mse = -1
    best_pred_mat = np.empty(())
    manualSeed = [random.randint(1, 99999999) for i in range(train_epochs)]

    for epoch in range(train_epochs):
        if verbose:
            print("train epoch: ", epoch)

        running_loss = 0.0

        #reload loader for shuffle
        #batch samplers used to draw samples in dataloaders
        batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)
        batch_sampler_all = pytorch_data_operations.RandomContiguousBatchSampler(all_data.size()[0], seq_length, yhat_batch_size, n_batches)

        alldataloader = DataLoader(total_data, batch_sampler=batch_sampler_all, pin_memory=True)
        trainloader = DataLoader(train_data, batch_sampler=batch_sampler, pin_memory=True)
        multi_loader = pytorch_data_operations.MultiLoader([trainloader, alldataloader])


        #zero the parameter gradients
        optimizer.zero_grad()
        lstm_net.train(True)
        avg_loss = 0
        batches_done = 0
        pdb.set_trace()
        for i, batches in enumerate(multi_loader):
            #load data
            inputs = None
            targets = None
            for j, b in enumerate(batches):
                if j == 0:
                    inputs, targets = b


            #cuda commands
            if(use_gpu):
                inputs = inputs.cuda()
                targets = targets.cuda()

            #forward  prop
            lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            h_state = None
            outputs, h_state = lstm_net(inputs, h_state)
            outputs = outputs.view(outputs.size()[0],-1)

            #calculate losses
            reg1_loss = 0
            if lambda1 > 0:
                reg1_loss = calculate_l1_loss(lstm_net)


            loss_outputs = outputs[:,begin_loss_ind:]
            loss_targets = targets[:,begin_loss_ind:].cpu()


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
            avg_unsup_loss += unsup_loss
            batches_done += 1

        #check for convergence
        avg_loss = avg_loss / batches_done
        if verbose:
            print("rmse loss=", avg_loss)

        if epoch % 50 == 0 and epoch != 0:
            save_path = "../../models/"+lakename+"/LSTM_source_model_"+str(n_hidden)+"hid_"+str(epoch)+"ep"
            saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
            print("saved at ",save_path)


