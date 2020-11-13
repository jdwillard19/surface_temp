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
save = True

#RMSE threshold for pretraining
num_layers = 1



#####################3
#params
###########################33
first_save_epoch = 0
patience = 100

n_hidden_list = [20,50] #fixed

unsup_loss_cutoff = 40
dc_unsup_loss_cutoff = 1e-3
dc_unsup_loss_cutoff2 = 1e-2
#ow
seq_length = 350 #how long of sequences to use in model
begin_loss_ind = 0#index in sequence where we begin to calculate error or predict
n_features = 7  #number of physical drivers
win_shift = 50 #how much to slide the window on training set each time
save = True 
grad_clip = 1.0 #how much to clip the gradient 2-norm in training


n_eps = 300

ep_list20 = [] #list of epochs at which models were saved for 20 hidden units
ep_list50 = [] #list of epochs at which models were saved for 50 hidden units

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
    patience = 50
    epoch_since_best = 0
    lambda1 = 0.0001
    data_dir = "../../data/processed/"+lakename+"/"
    yhat_batch_size = 1

    ###############################
    # data preprocess
    ##################################
    #create train and test sets

    (trn_data, all_data, all_phys_data, all_dates) = buildLakeDataForRNNPretrain(lakename, data_dir, seq_length, n_features,
                                       win_shift= win_shift, begin_loss_ind=begin_loss_ind,
                                       excludeTest=False, normAll=False, normGE10=False)
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
    # testloader = torch.utils.data.DataLoader(tst_data, batch_size=tst_data.size()[0], shuffle=False, pin_memory=True)


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
    ep_since_min = 0
    best_pred_mat = np.empty(())
    manualSeed = [random.randint(1, 99999999) for i in range(n_eps)]

    #stop training if true
    done = False

    for epoch in range(n_eps):
        if verbose:
            print("train epoch: ", epoch)

        if done:
            break

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
            batches_done += 1

        #check for convergence
        avg_loss = avg_loss / batches_done
        if verbose:
            print("rmse loss=", avg_loss)

        if avg_loss < min_mse:
            min_mse = avg_loss
            ep_min_mse = epoch
            ep_since_min = 0

        else:
            ep_since_min += 1

        if ep_since_min == patience:
            print("patience met")
            done = True
            break




        if epoch % 100 == 0 and epoch != 0:

            save_path = "../../models/"+lakename+"/LSTM_source_model_"+str(n_hidden)+"hid_"+str(epoch)+"ep"

            saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
            if n_hidden == 20:
                ep_list20.append(epoch)
            elif n_hidden == 50:
                ep_list50.append(epoch)

            print("saved at ",save_path)



print("|\n|\nTraining Candidate Models Complete\n|\n|")
pdb.set_trace()
##################################################################
# transfer all models to all other source lakes to find best one
########################################################################

#load all other source lakes
glm_all_f = pd.read_csv("../../results/glm_transfer/RMSE_transfer_glm_pball.csv")
train_lakes = np.array([re.search('nhdhr_(.*)', x).group(1) for x in np.unique(glm_all_f['target_id'].values)])

other_source_ids = train_lakes[~np.isin(train_lakes,site_id)] #remove site id
other_source_ids = other_source_ids[~np.isin(other_source_ids, ['121623043','121623126',\
                                                                '121860894','143249413',\
                                                                '143249864', '152335372',\
                                                                '155635994','70332223',\
                                                                '75474779'])] #remove cuz <= 1 surf temp obs


err_per_epoch20 = np.empty((len(ep_list20)))
err_per_epoch20[:] = np.nan

err_per_epoch50 = np.empty((len(ep_list50)))
err_per_epoch50[:] = np.nan


top_ids = [site_id]

#data structs to record transfer test results
err_per_hid_ep20 = np.empty((len(ep_list20)))
err_per_hid_ep50 = np.empty((len(ep_list50)))

for hid_ct, n_hidden in enumerate(n_hidden_list):
    ep_list = []
    if n_hidden == 20:
        ep_list = ep_list20
    elif n_hidden == 50:
        ep_list = ep_list50

    for ep_ct, eps in enumerate(ep_list):
        for target_id in other_source_ids:
            print("TARGET: ", target_id)
            data_dir_target = "../../data/processed/"+target_id+"/" 
            #target agnostic model and data params
            use_gpu = True
            n_features = 7
            # n_hidden = 20
            seq_length = 350
            win_shift = 175
            begin_loss_ind = 0
            (_, _, tst_data_target, tst_dates_target, unique_tst_dates_target, all_data_target, \
             all_phys_data_target, all_dates_target)\
            = buildLakeDataForRNN_manylakes_finetune2(target_id, data_dir_target, seq_length, n_features,
                                               win_shift = win_shift, begin_loss_ind = begin_loss_ind, 
                                               outputFullTestMatrix=True, allTestSeq=True)
            

            #useful values, LSTM params
            batch_size = all_data_target.size()[0]
            n_test_dates_target = unique_tst_dates_target.shape[0]


            #define LSTM model
            class LSTM(nn.Module):
                def __init__(self, input_size, hidden_size, batch_size):
                    super(LSTM, self).__init__()
                    self.input_size = input_size
                    self.hidden_size = hidden_size
                    self.batch_size = batch_size
                    self.lstm = nn.LSTM(input_size = n_features, hidden_size=hidden_size, batch_first=True, num_layers=num_layers) 
                    self.out = nn.Linear(hidden_size, 1)
                    self.hidden = self.init_hidden()

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
                
                def forward(self, x, hidden): #forward network propagation 
                    self.lstm.flatten_parameters()
                    x = x.float()
                    x, hidden = self.lstm(x, self.hidden)
                    self.hidden = hidden
                    x = self.out(x)
                    return x, hidden



            #output matrix
            n_lakes = len(top_ids)
            output_mats = np.empty((n_lakes, n_test_dates_target))
            ind_rmses = np.empty((n_lakes))
            ind_rmses[:] = np.nan
            label_mats = np.empty((n_test_dates_target)) 
            output_mats[:] = np.nan
            label_mats[:] = np.nan


            for i, source_id in enumerate(top_ids): 
                #for each top id
                # source_id = re.search('nhdhr_(.*)', source_id).group(1)
                #load source model
                load_path = "../../models/"+source_id+"/LSTM_source_model_"+str(n_hidden)+"hid_"+str(eps)+"ep"
                lstm_net = LSTM(n_features, n_hidden, batch_size)
                if use_gpu:
                    lstm_net = lstm_net.cuda(0)
                pretrain_dict = torch.load(load_path)['state_dict']
                model_dict = lstm_net.state_dict()
                pretrain_dict = {key: v for key, v in pretrain_dict.items() if key in model_dict}
                model_dict.update(pretrain_dict)
                lstm_net.load_state_dict(pretrain_dict)

                #things needed to predict test data
                mse_criterion = nn.MSELoss()
                testloader = torch.utils.data.DataLoader(tst_data_target, batch_size=tst_data_target.size()[0], shuffle=False, pin_memory=True)

                lstm_net.eval()
                with torch.no_grad():
                    avg_mse = 0
                    ct = 0
                    for m, data in enumerate(testloader, 0):
                        #now for mendota data
                        #this loop is dated, there is now only one item in testloader

                        #parse data into inputs and targets
                        inputs = data[:,:,:n_features].float()
                        targets = data[:,:,-1].float()
                        targets = targets[:, begin_loss_ind:]
                        tmp_dates = tst_dates_target[:, begin_loss_ind:]
                        depths = inputs[:,:,0]

                        if use_gpu:
                            inputs = inputs.cuda()
                            targets = targets.cuda()

                        #run model
                        h_state = None
                        lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
                        pred, h_state = lstm_net(inputs, h_state)
                        pred = pred.view(pred.size()[0],-1)
                        pred = pred[:, begin_loss_ind:]

                        #calculate error
                        targets = targets.cpu()
                        loss_indices = np.where(~np.isnan(targets))
                        if use_gpu:
                            targets = targets.cuda()
                        inputs = inputs[:, begin_loss_ind:, :]
                        depths = depths[:, begin_loss_ind:]
                        mse = mse_criterion(pred[loss_indices], targets[loss_indices])
                        # print("test loss = ",mse)
                        avg_mse += mse

                        if mse > 0: #obsolete i think
                            ct += 1
                        avg_mse = avg_mse / ct


                        #save model 
                        (outputm_npy, labelm_npy) = parseMatricesFromSeqs(pred.cpu().numpy(), targets.cpu().numpy(), tmp_dates, 
                                                                        n_test_dates_target,
                                                                        unique_tst_dates_target) 
                        #to store output
                        output_mats[i,:] = outputm_npy
                        if i == 0:
                            #store label
                            label_mats = labelm_npy
                        loss_output = outputm_npy[~np.isnan(labelm_npy)]
                        loss_label = labelm_npy[~np.isnan(labelm_npy)]

                        mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())
                        # print(source_id+" rmse=", mat_rmse)

                        # glm_rmse = float(metadata.loc["nhdhr_"+target_id].glm_uncal_rmse_full)

                        # mat_csv.append(",".join(["nhdhr_"+target_id,"nhdhr_"+ source_id,str(meta_rmse_per_lake[targ_ct]),str(srcorr_per_lake[targ_ct]), str(glm_rmse),str(mat_rmse)] + [str(x) for x in lake_df.iloc[i][feats].values]))


            #save model 
            total_output_npy = np.average(output_mats, axis=0)

            # if output_to_file:
            #     outputm_npy = np.transpose(total_output_npy)
            #     label_mat= np.transpose(label_mats)
            #     output_df = pd.DataFrame(data=outputm_npy, columns=[str(float(x/2)) for x in range(outputm_npy.shape[1])], index=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
            #     label_df = pd.DataFrame(data=label_mat, columns=[str(float(x/2)) for x in range(label_mat.shape[1])], index=[str(x)[:10] for x in unique_tst_dates_target]).reset_index()
            #     output_df.rename(columns={'index': 'depth'})
            #     label_df.rename(columns={'index': 'depth'})

            #     assert np.isfinite(np.array(output_df.values[:,1:],dtype=np.float32)).all(), "nan output"
            #     lake_output_path = output_path+target_id
            #     if not os.path.exists(lake_output_path):
            #         os.mkdir(lake_output_path)
            #     output_df.to_feather(lake_output_path+"/PGMTL_outputs.feather")
                
            loss_output = total_output_npy[~np.isnan(label_mats)]
            loss_label = label_mats[~np.isnan(label_mats)]
            mat_rmse = np.sqrt(((loss_output - loss_label) ** 2).mean())

            print("source ",site_id, "-> target ", target_id,": Total rmse=", mat_rmse)

            if n_hidden == 20:
                err_per_hid_ep20[ep_ct] = mat_rmse
            elif n_hidden == 50:
                err_per_hid_ep50[ep_ct] = mat_rmse





pdb.set_trace()
with open(save_file_path,'w') as file:
    for line in mat_csv:
        file.write(line)
        file.write('\n')


