import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np
import sys
sys.path.append('../data')
import pytorch_data_operations
import pandas as pd
import pdb
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_ 
from pytorch_data_operations import buildDataTCN_trn

########################################################
# July 2021 - trying out new arch
#######################################################

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



#load data?
# trn_data_path = "../../../lake_conus_surface_temp_2021/src/evaluate/ealstm_trn_data_062421_5fold_k1.npy"
# trn_data = torch.from_numpy(np.load(trn_data_path))

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


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])

tcn = TCN(input_size=n_features, output_size=1,num_channels=n_features+1,kernel_size=2,dropout=.2)

if torch.cuda.is_available():
    tcn = tcn.cuda()



#load metadata
metadata = pd.read_csv("../../../lake_conus_surface_temp_2021/metadata/lake_metadata.csv")

#trim to observed lakes
metadata = metadata[metadata['num_obs'] > 0]

lakenames = metadata['site_id'].values[:100]
seq_length = 365
n_features = 9
#create training data

(trn_data,trn_dates) = buildDataTCN_trn(lakenames, seq_length,n_features)

#format training data for loading
train_data = TemperatureTrainDataset(trn_data)

#params
batch_size = 10
n_eps = 1000
n_batches = math.floor(trn_data.size()[0] / batch_size)
optimizer = torch.optim.AdamW(trn.parameters(), lr=lr)

for epoch in range(n_eps):
    if done:
        break
    # if verbose and epoch % 10 == 0:
    if verbose:
        print("train epoch: ", epoch)

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
        # targets = targets[:, begin_loss_ind:]
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
    train_avg_loss = avg_loss


    if verbose:
        print("train rmse loss=", avg_loss)

    if avg_loss < min_train_rmse:
        min_train_rmse = avg_loss
        print("model saved")
        save_path = "../../models/EALSTM_"+str(n_hidden)+"hid_"+str(num_layers)+"_final_070221"
        saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)

    if avg_loss < targ_rmse and epoch > targ_ep:
        print("training complete")
        break

    # #each epoch do these
    # batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)

    # trainloader = DataLoader(train_data, batch_sampler=batch_sampler, pin_memory=True)


    # #zero the parameter gradients
    # optimizer.zero_grad()
    # lstm_net.train(True)
    # avg_loss = 0
    # batches_done = 0
    # ct = 0
    # for m, data in enumerate(trainloader, 0):
    #     #now for mendota data
    #     #this loop is dated, there is now only one item in testloader

    #     #parse data into inputs and targets
    #     inputs = data[0].float()
    #     targets = data[1].float()
    #     targets = targets[:, begin_loss_ind:]





