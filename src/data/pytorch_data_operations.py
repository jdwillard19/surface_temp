import torch
import numpy as np
import pandas as pd
import math
import sys
import phys_operations
import datetime
from datetime import date
import os
import torch.nn as nn
from torch.nn.init import xavier_normal_
import pdb
from scipy import interpolate


def buildLakeDataForRNN_multilakemodel(lakenames, seq_length, n_features, \
                                            win_shift= 1, begin_loss_ind = 100, \
                                            test_seq_per_depth=1, sparseCustom=None, \
                                            allTestSeq=False, verbose=False,\
                                            oldFeat = False, normGE10=False, postProcessSplits=True, randomSeed=0,static_feats=False,n_static_feats = 0):

    #NONAN
    #PARAMETERS
        #@lakenames = array of lake names as the folder of /data/processed/{lakename}
        #@seq_length = sequence length of LSTM inputs
        #@n_features = number of physical drivers
        #@win_shift = days to move in the sliding window for the training set
        #@begin_loss_ind = index in sequence to begin calculating loss function (to avoid poor accuracy in early parts of the sequence)

    #composite data structures
    n_features = n_features+n_static_feats
    X_trn_comp = torch.Tensor(0, seq_length, n_features+1)
    trn_dates_comp = torch.Tensor(0, seq_length)
    X_tst_comp = torch.Tensor(0, seq_length, n_features+1)
    tst_dates_comp = torch.Tensor(0, seq_length)
    # X_all_comp = torch.Tensor(0, seq_length, n_features+1)
    # X_phys_comp = torch.Tensor(0, seq_length, n_features+1)
    # all_dates_comp = torch.Tensor(0, seq_length)

    for lake_ct, lakename in enumerate(lakenames):
        if verbose:
            print("loading data for lake ",lake_ct,"/",len(lakenames))
        #load data created in preprocess.py based on lakename
        debug = False
        # verbose = True
        my_path = os.path.abspath(os.path.dirname(__file__))

        if not static_feats:
            feat_mat_raw = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/features.npy"))
            feat_mat = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/processed_features.npy"))
        else:
            feat_mat_raw = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/features_ea.npy"))
            feat_mat = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/processed_features_ea.npy"))


        tst = []
        trn = []

        #GET TRAIN/TEST HERE
        #alternative way to divide test/train just by 1/3rd 2/3rd
        # tst = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/test.npy"))
        # trn = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/train.npy"))
        if os.path.exists(os.path.join(my_path, "../../data/processed/"+lakename+"/full.npy")):
            full = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/full.npy"))
        else:
            full = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/full_obs.npy"))



        # if debug:
        #     print("initial trn: ", trn)
        #     print("observations: ",np.count_nonzero(~np.isnan(trn)))
        dates = []
        if os.path.exists(os.path.join(my_path, "../../data/processed/"+lakename+"/dates.npy")):
            dates = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/dates.npy"))
        else:
            dates = []

        #post process train/test splits

        # if postProcessSplits:
        #     shape0 = trn.shape[0]
        #     trn_flt = trn.flatten()
        #     tst_flt = tst.flatten()
        #     np.put(trn_flt, np.where(np.isfinite(tst_flt))[0], tst_flt[np.isfinite(tst_flt)])
        #     trn_tst = trn_flt.reshape((shape0))
        #     last_tst_col = int(np.round(np.unique(np.where(np.isfinite(trn_tst))).shape[0]/3))
        #     unq_col = np.unique(np.where(np.isfinite(trn_tst))[1])
        #     trn = np.empty_like(trn_tst)
        #     trn[:] = np.nan
        #     tst = np.empty_like(trn_tst)
        #     tst[:] = np.nan
        #     trn[:,unq_col[last_tst_col]:] = trn_tst[:,unq_col[last_tst_col]:]
        #     tst[:,:unq_col[last_tst_col]] = trn_tst[:,:unq_col[last_tst_col]]

        np.random.seed(seed=randomSeed)
        if sparseCustom is not None:
            n_profiles = np.nonzero(np.count_nonzero(~np.isnan(trn), axis=0))[0].shape[0] #n nonzero columns
            n_profiles_to_zero = n_profiles - sparseCustom #n nonzero columns
            if n_profiles_to_zero < 0:
                print("not enough training obs")
                return((sparseCustom,None,None,None,None,None,None,None,None))
            profiles_ind = np.nonzero(np.count_nonzero(~np.isnan(trn), axis=0))[0] #nonzero columns
            index = np.random.choice(profiles_ind.shape[0], n_profiles_to_zero, replace=False)  
            trn[:,profiles_ind[index]] = np.nan
            # tst = trn
        #convert dates to numpy datetime64
        # dates = [date.decode() for date in dates]
        dates = pd.to_datetime(dates, format='%Y-%m-%d')
        dates = np.array(dates,dtype=np.datetime64)

        years = dates.astype('datetime64[Y]').astype(int) + 1970
        assert np.isfinite(feat_mat).all(), "feat_mat has nan at" + str(np.argwhere(np.isfinite(feat_mat)))
        assert np.isfinite(feat_mat_raw).all(), "feat_mat_raw has nan at" + str(np.argwhere(np.isfinite(feat_mat_raw)))
        # assert np.isfinite(Y_mat).any(), "Y_mat has nan at" + str(np.argwhere(np.isfinite(Y_mat)))

        n_depths = feat_mat.shape[0]
        assert feat_mat.shape[0] == feat_mat_raw.shape[0]
        assert feat_mat.shape[0] == tst.shape[0]
        assert feat_mat.shape[0] == trn.shape[0]
        win_shift_tst = begin_loss_ind
        udates = dates
        n_dates = feat_mat.shape[0]
        if verbose:
            print("n dates: ", n_dates)
        seq_per_depth = math.floor(n_dates / seq_length)
        train_seq_per_depth = seq_per_depth
        test_seq_per_depth = seq_per_depth
        win_per_seq = math.floor(seq_length / win_shift) - 1 #windows per sequence (only training)
        win_per_seq = 2#windows per sequence (only training)
        tst_win_per_seq = 2 #windows per sequence (only training)
        n_train_seq = train_seq_per_depth * win_per_seq
        if n_dates % seq_length > 0 and n_dates - seq_length > 0:
            n_train_seq += 1

        if debug:
            print("n train seq: ", n_train_seq)

        n_train_seq_no_window = train_seq_per_depth 

        last_test_date_ind = np.where(np.isfinite(tst))[0][-1]
        n_test_seq = (test_seq_per_depth) * tst_win_per_seq
        if last_test_date_ind % seq_length > 0 and last_test_date_ind - seq_length > 0:
            n_test_seq += 1




        #build train and test sets, add all data for physical loss

        X_trn = np.empty(shape=(n_train_seq, seq_length, n_features+1)) #features + label
        X_tst = np.empty(shape=(n_test_seq, seq_length, n_features+1))
        trn_dates = np.empty(shape=(n_train_seq, seq_length), dtype='datetime64[s]')
        tst_dates = np.empty(shape=(n_test_seq, seq_length), dtype='datetime64[s]')
        trn_dates[:] = np.datetime64("NaT")
        tst_dates[:] = np.datetime64("NaT")


        X_trn[:] = np.nan
        X_tst[:] = np.nan

        #seq index for data to be returned
        tr_seq_ind = 0 
        ts_seq_ind = 0
        #build datasets
        del_all_seq = 0
        if debug:
            print("x_trn shape prior to populating ", X_trn.shape)
        # print("obs index: ", np.where(np.isfinite(trn)))
        for s in range(0,train_seq_per_depth):
            start_index = s*seq_length
            end_index = (s+1)*seq_length
            if end_index > n_dates:
                n_train_seq -= win_per_seq*n_depths
                X_trn = np.delete(X_trn, np.arange(X_trn.shape[0],X_trn.shape[0]-win_per_seq*n_depths,-1), axis=0)
                trn_dates = np.delete(trn_dates, np.arange(trn_dates.shape[0], trn_dates.shape[0] - n_depths*win_per_seq,-1), axis=0)
                continue

            #now do sliding windows for training data 
            for w in range(0, win_per_seq):
                win_start_ind = start_index + w*win_shift
                win_end_ind = win_start_ind + seq_length


                if win_end_ind > n_dates:
                    n_train_seq -= 1
                    X_trn = np.delete(X_trn, -1, axis=0)
                    trn_dates = np.delete(trn_dates, -1, axis=0)
                    continue
                X_trn[tr_seq_ind, :, :-1] = feat_mat[win_start_ind:win_end_ind,:]
                X_trn[tr_seq_ind,:,-1] = trn[win_start_ind:win_end_ind]
                trn_dates[tr_seq_ind,:] = dates[win_start_ind:win_end_ind]
                tr_seq_ind += 1
        #final seq starts at end and goes inward [seq_length]
        # print("n dates: ", n_dates, ", seq len: ", seq_length)
        if n_dates % seq_length > 0:
            end_ind = n_dates
            start_ind = end_ind - seq_length
            X_trn[tr_seq_ind, :, :-1] = feat_mat[start_ind:end_ind,:]
            X_trn[tr_seq_ind,:,-1] = trn[start_ind:end_ind]
            trn_dates[tr_seq_ind,:] = dates[start_ind:end_ind]
            tr_seq_ind += 1

        if debug:
            print("x_trn shape after populating ", X_trn.shape)
        #assert data was constructed correctly
        if tr_seq_ind != n_train_seq:
            # print("incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind))
            extra = n_train_seq - tr_seq_ind
            n_train_seq -= extra
            X_trn = np.delete(X_trn, np.arange(X_trn.shape[0],X_trn.shape[0]-extra,-1), axis=0)
            trn_dates = np.delete(trn_dates, np.arange(X_trn.shape[0],X_trn.shape[0]-extra,-1), axis=0)
        assert tr_seq_ind == n_train_seq, \
         "incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind)

        while trn_dates[-1,0] == np.datetime64("NaT"):
            trn_dates = np.delete(trn_dates, -1, axis=0)
            X_trn = np.delete(X_trn, -1, axis=0)

        if n_test_seq != 0:
            #now test data(maybe bug in this specification of end of range?)
            for s in range(test_seq_per_depth):
                    start_index = s*seq_length
                    end_index = (s+1)*seq_length
                    if end_index > n_dates:
                        n_test_seq -= tst_win_per_seq
                        X_tst = np.delete(X_tst, np.arange(X_tst.shape[0], X_tst.shape[0] - tst_win_per_seq,-1), axis=0)
                        tst_dates = np.delete(tst_dates, np.arange(tst_dates.shape[0], tst_dates.shape[0] - tst_win_per_seq,-1), axis=0)
                        continue
                    for w in range(0, tst_win_per_seq):
                        win_start_ind = start_index+w*win_shift_tst
                        win_end_ind = win_start_ind + seq_length
                        if win_end_ind > n_dates:
                            n_test_seq -= 1
                            X_tst = np.delete(X_tst, np.arange(X_tst.shape[0], X_tst.shape[0] - 1,-1), axis=0)
                            tst_dates = np.delete(tst_dates, np.arange(tst_dates.shape[0], tst_dates.shape[0] - 1,-1), axis=0)
                            continue

                        X_tst[ts_seq_ind, :, :-1] = feat_mat[win_start_ind:win_end_ind,:]
                        X_tst[ts_seq_ind,:,-1] = tst[win_start_ind:win_end_ind]
                        tst_dates[ts_seq_ind,:] = dates[win_start_ind:win_end_ind]
                        ts_seq_ind += 1
                               #final seq starts at end and goes inward [seq_length]
            if debug:
                print("last_test_date_ind: ", last_test_date_ind, ", sl ", seq_length)

            if last_test_date_ind % seq_length > 0 and last_test_date_ind - seq_length > 0:
                end_ind = last_test_date_ind
                if not end_ind - seq_length < 0:
                    start_ind = end_ind - seq_length

                    X_tst[ts_seq_ind, :, :-1] = feat_mat[start_ind:end_ind,:]
                    X_tst[ts_seq_ind,:,-1] = trn[start_ind:end_ind]
                    tst_dates[ts_seq_ind,:] = dates[start_ind:end_ind]
                    ts_seq_ind += 1
        while np.isnat(tst_dates[-1,0]):
            if debug:
                print("NaT?")
            tst_dates = np.delete(tst_dates, -1, axis=0)
            X_tst = np.delete(X_tst, -1, axis=0)




        #assert data was constructed correctly
        assert ts_seq_ind == n_test_seq, \
            "incorrect number of tst seq estimated {} vs actual{}".format(n_test_seq, ts_seq_ind)      
        #remove sequences with no labels
        tr_seq_removed = 0
        trn_del_ind = np.array([], dtype=np.int32)
        ts_seq_removed = 0
        tst_del_ind = np.array([], dtype=np.int32)

        #if allTestSeq, combine trn and test unfiltered
        if allTestSeq:
            X_tst = np.vstack((X_tst, X_trn))
            tst_dates = np.vstack((tst_dates, trn_dates))
            while np.isnat(tst_dates[-1,0]):
                tst_dates = np.delete(tst_dates, -1, axis=0)
                X_tst = np.delete(X_tst, -1, axis=0)
        # full_data = 
        for i in range(X_trn.shape[0]):
            # print("seq ",i," nz-val-count:",np.count_nonzero(~np.isnan(X_trn[i,:,-1])))
            if not np.isfinite(X_trn[i,:,:-1]).all():
                # print("MISSING FEAT REMOVE")
                tr_seq_removed += 1
                trn_del_ind = np.append(trn_del_ind, i)
            if np.isfinite(X_trn[i,begin_loss_ind:,-1]).any():
                # print("HAS OBSERVE, CONTINUE")
                continue
            else:
                # print(X_trn[i,:,-1])
                # print("NO OBSERVE, REMOVE")
                tr_seq_removed += 1
                trn_del_ind = np.append(trn_del_ind, i)
        if not allTestSeq: #if we don't want to output ALL the data as test
            for i in range(X_tst.shape[0]):
                if not np.isfinite(X_tst[i,:,:-1]).all():
                    ts_seq_removed += 1
                    tst_del_ind = np.append(tst_del_ind, i)
                if np.isfinite(X_tst[i,begin_loss_ind:,-1]).any():
                    continue
                else:
                    tst_del_ind = np.append(tst_del_ind, i)
                    ts_seq_removed += 1
                if i > 0:
                    if tst_dates[i-1,0] > tst_dates[i,0]:
                        if debug:
                            print("date thing?")
                        tst_del_ind = np.append(tst_del_ind, i)
        else:
            for i in range(X_tst.shape[0]):
                if not np.isfinite(X_tst[i,:,:-1]).all():
                    ts_seq_removed += 1
                    tst_del_ind = np.append(tst_del_ind, i)
        #remove denoted values from trn and tst
        X_trn_tmp = np.delete(X_trn, trn_del_ind, axis=0)
        trn_dates_tmp = np.delete(trn_dates, trn_del_ind, axis=0)
        X_tst_tmp = np.delete(X_tst, tst_del_ind, axis=0)
        tst_dates_tmp = np.delete(tst_dates, tst_del_ind, axis=0)
        X_trn = X_trn_tmp
        trn_dates = trn_dates_tmp
        X_tst = X_tst_tmp
        tst_dates = tst_dates_tmp
        #gather unique test dates
        tst_date_lower_bound = np.where(dates == tst_dates[0][0])[0][0]
        tst_date_upper_bound = np.where(dates == tst_dates[-1][-1])[0][0]



        assert np.isfinite(X_trn[:,:,:-1]).all(), "X_trn has nan"
        # assert np.isfinite(all_dates).any(), "all_dates has nan"
        X_trn_comp = torch.cat([X_trn_comp,torch.from_numpy(X_trn).float()],dim=0)
        # trn_dates_comp = torch.cat([trn_dates_comp,torch.from_numpy(trn_dates)],dim=0)
        X_tst_comp = torch.cat([X_tst_comp,torch.from_numpy(X_tst).float()],dim=0)
        # tst_dates_comp = torch.cat([tst_dates_comp,torch.from_numpy(tst_dates)],dim=0)
        
    return (X_trn_comp, trn_dates_comp, X_tst_comp, tst_dates_comp)

def buildLakeDataForRNN_multilakemodel_conus(lakenames, seq_length, n_features, \
                                            win_shift= 1, begin_loss_ind = 100, \
                                            test_seq_per_depth=1, sparseCustom=None, \
                                            allTestSeq=False,verbose=False,\
                                            oldFeat = False, normGE10=False, postProcessSplits=True, randomSeed=0,static_feats=False,n_static_feats = 0):

    #NONAN
    #PARAMETERS
        #@lakenames = array of lake names as the folder of /data/processed/{lakename}
        #@seq_length = sequence length of LSTM inputs
        #@n_features = number of physical drivers
        #@win_shift = days to move in the sliding window for the training set
        #@begin_loss_ind = index in sequence to begin calculating loss function (to avoid poor accuracy in early parts of the sequence)

    #composite data structures
    # n_features = n_features+n_static_feats
    X_trn_comp = torch.Tensor(0, seq_length, n_features+1)
    trn_dates_comp = torch.Tensor(0, seq_length)
    X_tst_comp = torch.Tensor(0, seq_length, n_features+1)
    tst_dates_comp = torch.Tensor(0, seq_length)
    # X_all_comp = torch.Tensor(0, seq_length, n_features+1)
    # X_phys_comp = torch.Tensor(0, seq_length, n_features+1)
    # all_dates_comp = torch.Tensor(0, seq_length)
    lakes_early_obs = []
    for lake_ct, lakename in enumerate(lakenames):
        if verbose:
            print("loading data for lake ",lake_ct,"/",len(lakenames))
        # if lake_ct < 217:
            # continue
        #load data created in preprocess.py based on lakename
        debug = False
        # verbose = True
        my_path = os.path.abspath(os.path.dirname(__file__))

        feat_mat_raw = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/features_ea_conus_021621.npy"))
        feat_mat = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/processed_features_ea_conus_021621.npy"))

        # if removeElevation:
        #     feat_mat = np.delete(feat_mat,29,axis=1)
        #     feat_mat_raw = np.delete(feat_mat_raw,29,axis=1)
        trn = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/full.npy"))



        dates = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/dates.npy"))

        #post process train/test splits

        # if postProcessSplits:
        #     shape0 = trn.shape[0]
        #     trn_flt = trn.flatten()
        #     tst_flt = tst.flatten()
        #     np.put(trn_flt, np.where(np.isfinite(tst_flt))[0], tst_flt[np.isfinite(tst_flt)])
        #     trn_tst = trn_flt.reshape((shape0))
        #     last_tst_col = int(np.round(np.unique(np.where(np.isfinite(trn_tst))).shape[0]/3))
        #     unq_col = np.unique(np.where(np.isfinite(trn_tst))[1])
        #     trn = np.empty_like(trn_tst)
        #     trn[:] = np.nan
        #     tst = np.empty_like(trn_tst)
        #     tst[:] = np.nan
        #     trn[:,unq_col[last_tst_col]:] = trn_tst[:,unq_col[last_tst_col]:]
        #     tst[:,:unq_col[last_tst_col]] = trn_tst[:,:unq_col[last_tst_col]]

        np.random.seed(seed=randomSeed)
        if sparseCustom is not None:
            n_profiles = np.nonzero(np.count_nonzero(~np.isnan(trn), axis=0))[0].shape[0] #n nonzero columns
            n_profiles_to_zero = n_profiles - sparseCustom #n nonzero columns
            if n_profiles_to_zero < 0:
                print("not enough training obs")
                return((sparseCustom,None,None,None,None,None,None,None,None))
            profiles_ind = np.nonzero(np.count_nonzero(~np.isnan(trn), axis=0))[0] #nonzero columns
            index = np.random.choice(profiles_ind.shape[0], n_profiles_to_zero, replace=False)  
            trn[:,profiles_ind[index]] = np.nan
            # tst = trn
        #convert dates to numpy datetime64
        # dates = [date.decode() for date in dates]
        dates = pd.to_datetime(dates, format='%Y-%m-%d')
        dates = np.array(dates,dtype=np.datetime64)

        years = dates.astype('datetime64[Y]').astype(int) + 1970
        assert np.isfinite(feat_mat).all(), "feat_mat has nan at" + str(np.argwhere(np.isfinite(feat_mat)))
        assert np.isfinite(feat_mat_raw).all(), "feat_mat_raw has nan at" + str(np.argwhere(np.isfinite(feat_mat_raw)))
        # assert np.isfinite(Y_mat).any(), "Y_mat has nan at" + str(np.argwhere(np.isfinite(Y_mat)))

        # n_depths = feat_mat.shape[0]
        assert feat_mat.shape[0] == feat_mat_raw.shape[0]
        # assert feat_mat.shape[0] == tst.shape[0]
        assert feat_mat.shape[0] == trn.shape[0]
        win_shift_tst = begin_loss_ind
        udates = dates
        n_dates = feat_mat.shape[0]
        if verbose:
            print("n dates: ", n_dates)
        seq_per_depth = math.floor(n_dates / seq_length)
        train_seq_per_depth = seq_per_depth
        # test_seq_per_depth = seq_per_depth
        win_per_seq = math.floor(seq_length / win_shift) - 1 #windows per sequence (only training)
        win_per_seq = 2#windows per sequence (only training)
        # tst_win_per_seq = 2 #windows per sequence (only training)
        n_train_seq = train_seq_per_depth * win_per_seq
        if n_dates % seq_length > 0 and n_dates - seq_length > 0:
            n_train_seq += 1

        if debug:
            print("n train seq: ", n_train_seq)

        n_train_seq_no_window = train_seq_per_depth 

        # last_test_date_ind = np.where(np.isfinite(tst))[0][-1]
        # n_test_seq = (test_seq_per_depth) * tst_win_per_seq
        # if last_test_date_ind % seq_length > 0 and last_test_date_ind - seq_length > 0:
            # n_test_seq += 1




        #build train and test sets, add all data for physical loss

        X_trn = np.empty(shape=(n_train_seq, seq_length, n_features+1)) #features + label
        # X_tst = np.empty(shape=(n_test_seq, seq_length, n_features+1))
        trn_dates = np.empty(shape=(n_train_seq, seq_length), dtype='datetime64[s]')
        # tst_dates = np.empty(shape=(n_test_seq, seq_length), dtype='datetime64[s]')
        trn_dates[:] = np.datetime64("NaT")
        # tst_dates[:] = np.datetime64("NaT")


        X_trn[:] = np.nan
        # X_tst[:] = np.nan

        #seq index for data to be returned
        tr_seq_ind = 0 
        # ts_seq_ind = 0
        #build datasets
        del_all_seq = 0
        if debug:
            print("x_trn shape prior to populating ", X_trn.shape)
        # print("obs index: ", np.where(np.isfinite(trn)))
        for s in range(0,train_seq_per_depth):
            start_index = s*seq_length
            end_index = (s+1)*seq_length
            if end_index > n_dates:
                n_train_seq -= win_per_seq*n_depths
                X_trn = np.delete(X_trn, np.arange(X_trn.shape[0],X_trn.shape[0]-win_per_seq*n_depths,-1), axis=0)
                trn_dates = np.delete(trn_dates, np.arange(trn_dates.shape[0], trn_dates.shape[0] - n_depths*win_per_seq,-1), axis=0)
                continue

            #now do sliding windows for training data 
            for w in range(0, win_per_seq):
                win_start_ind = start_index + w*win_shift
                win_end_ind = win_start_ind + seq_length


                if win_end_ind > n_dates:
                    n_train_seq -= 1
                    X_trn = np.delete(X_trn, -1, axis=0)
                    trn_dates = np.delete(trn_dates, -1, axis=0)
                    continue
                X_trn[tr_seq_ind, :, :-1] = feat_mat[win_start_ind:win_end_ind,:]
                X_trn[tr_seq_ind,:,-1] = trn[win_start_ind:win_end_ind]
                trn_dates[tr_seq_ind,:] = dates[win_start_ind:win_end_ind]
                tr_seq_ind += 1
        #final seq starts at end and goes inward [seq_length]
        # print("n dates: ", n_dates, ", seq len: ", seq_length)
        if n_dates % seq_length > 0:
            end_ind = n_dates
            start_ind = end_ind - seq_length
            if start_ind < 0:
                print("too early obs")
                lakes_early_obs.append(lakename)
                continue
            X_trn[tr_seq_ind, :, :-1] = feat_mat[start_ind:end_ind,:]
            X_trn[tr_seq_ind,:,-1] = trn[start_ind:end_ind]
            trn_dates[tr_seq_ind,:] = dates[start_ind:end_ind]
            tr_seq_ind += 1

        if debug:
            print("x_trn shape after populating ", X_trn.shape)
        #assert data was constructed correctly
        if tr_seq_ind != n_train_seq:
            # print("incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind))
            extra = n_train_seq - tr_seq_ind
            n_train_seq -= extra
            X_trn = np.delete(X_trn, np.arange(X_trn.shape[0],X_trn.shape[0]-extra,-1), axis=0)
            trn_dates = np.delete(trn_dates, np.arange(X_trn.shape[0],X_trn.shape[0]-extra,-1), axis=0)
        assert tr_seq_ind == n_train_seq, \
         "incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind)

        while trn_dates[-1,0] == np.datetime64("NaT"):
            trn_dates = np.delete(trn_dates, -1, axis=0)
            X_trn = np.delete(X_trn, -1, axis=0)


        tr_seq_removed = 0
        trn_del_ind = np.array([], dtype=np.int32)

        for i in range(X_trn.shape[0]):
            # print("seq ",i," nz-val-count:",np.count_nonzero(~np.isnan(X_trn[i,:,-1])))
            if not np.isfinite(X_trn[i,:,:-1]).all():
                # print("MISSING FEAT REMOVE")
                tr_seq_removed += 1
                trn_del_ind = np.append(trn_del_ind, i)
            if np.isfinite(X_trn[i,begin_loss_ind:,-1]).any():
                # print("HAS OBSERVE, CONTINUE")
                continue
            else:
                # print(X_trn[i,:,-1])
                # print("NO OBSERVE, REMOVE")
                tr_seq_removed += 1
                trn_del_ind = np.append(trn_del_ind, i)
        # if not allTestSeq: #if we don't want to output ALL the data as test
        #     for i in range(X_tst.shape[0]):
        #         if not np.isfinite(X_tst[i,:,:-1]).all():
        #             ts_seq_removed += 1Z
        #             tst_del_ind = np.append(tst_del_ind, i)
        #         if np.isfinite(X_tst[i,begin_loss_ind:,-1]).any():
        #             continue
        #         else:
        #             tst_del_ind = np.append(tst_del_ind, i)
        #             ts_seq_removed += 1
        #         if i > 0:
        #             if tst_dates[i-1,0] > tst_dates[i,0]:
        #                 if debug:
        #                     print("date thing?")
        #                 tst_del_ind = np.append(tst_del_ind, i)
        # else:
        #     for i in range(X_tst.shape[0]):
        #         if not np.isfinite(X_tst[i,:,:-1]).all():
        #             ts_seq_removed += 1
        #             tst_del_ind = np.append(tst_del_ind, i)
        #remove denoted values from trn and tst
        X_trn_tmp = np.delete(X_trn, trn_del_ind, axis=0)
        trn_dates_tmp = np.delete(trn_dates, trn_del_ind, axis=0)
        # X_tst_tmp = np.delete(X_tst, tst_del_ind, axis=0)
        # tst_dates_tmp = np.delete(tst_dates, tst_del_ind, axis=0)
        X_trn = X_trn_tmp
        trn_dates = trn_dates_tmp
        # X_tst = X_tst_tmp
        # tst_dates = tst_dates_tmp
        #gather unique test dates
        # tst_date_lower_bound = np.where(dates == tst_dates[0][0])[0][0]
        # tst_date_upper_bound = np.where(dates == tst_dates[-1][-1])[0][0]



        assert np.isfinite(X_trn[:,:,:-1]).all(), "X_trn has nan"
        # assert np.isfinite(all_dates).any(), "all_dates has nan"
        X_trn_comp = torch.cat([X_trn_comp,torch.from_numpy(X_trn).float()],dim=0)
        # trn_dates_comp = torch.cat([trn_dates_comp,torch.from_numpy(trn_dates)],dim=0)
        # X_tst_comp = torch.cat([X_tst_comp,torch.from_numpy(X_tst).float()],dim=0)
    #     # tst_dates_comp = torch.cat([tst_dates_comp,torch.from_numpy(tst_dates)],dim=0)
    # print(repr(lakes_early_obs))
    # print(len(lakes_early_obs))
    return (X_trn_comp, None)





def buildLakeDataForRNN_manylakes_gauged(lakenames, seq_length, n_features, \
                                            win_shift= 1, begin_loss_ind = 100, \
                                            outputFullTestMatrix=False, sparseCustom=None, \
                                            allTestSeq=False, static_feats=False,n_static_feats=0,\
                                            postProcessSplits=True, randomSeed=0):

    #NONAN
    #PARAMETERS
        #@lakename = string of lake name as the folder of /data/processed/{lakename}
        #@seq_length = sequence length of LSTM inputs
        #@n_features = number of physical drivers
        #@win_shift = days to move in the sliding window for the training set
        #@begin_loss_ind = index in sequence to begin calculating loss function (to avoid poor accuracy in early parts of the sequence)
    #load data created in preprocess.py based on lakename
    n_features = n_features+n_static_feats
    debug = False
    verbose = False
    my_path = os.path.abspath(os.path.dirname(__file__))

    #structs to fill as we loop through lakes
    X_trn_comp = torch.Tensor(0, seq_length, n_features+1)
    trn_dates_comp = np.array(torch.Tensor(0, seq_length),dtype=np.datetime64)
    X_tst_comp = torch.Tensor(0, seq_length, n_features+1)
    tst_dates_comp = np.array(torch.Tensor(0, seq_length),dtype=np.datetime64)

    # X_all_comp = torch.Tensor(0, seq_length, n_features+1)
    # X_phys_comp = torch.Tensor(0, seq_length, n_features+1)
    # all_dates_comp = torch.Tensor(0, seq_length)
    lakes_early_obs = []
    for lake_ct, lakename in enumerate(lakenames):
        feat_mat_raw = None
        feat_mat = None
        if static_feats:
            feat_mat_raw = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/features_ea_conus_021621.npy"))
            feat_mat = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/processed_features_ea_conus_021621.npy"))
        else:
            feat_mat_raw = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/features.npy"))
            feat_mat = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/processed_features.npy"))

        tst = []
        trn = []

        #GET TRAIN/TEST HERE
        #alternative way to divide test/train just by 1/3rd 2/3rd
        if os.path.exists(os.path.join(my_path, "../../data/processed/"+lakename+"/full.npy")):
            full = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/full.npy"))
        else:
            full = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/full_obs.npy"))


        dates = []
        if os.path.exists(os.path.join(my_path, "../../data/processed/"+lakename+"/dates.npy")):
            dates = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/dates.npy"))
        else:
            dates = []

        #post process train/test splits
        if postProcessSplits:
            # shape0 = trn.shape[0]
            # trn_flt = trn.flatten()
            # tst_flt = tst.flatten()
            # np.put(trn_flt, np.where(np.isfinite(tst_flt))[0], tst_flt[np.isfinite(tst_flt)])
            trn_tst = full.flatten()
            # trn_tst = trn_flt.reshape((shape0))
            last_tst_col = int(np.round(np.unique(np.where(np.isfinite(trn_tst))).shape[0]/3))
            unq_col = np.where(np.isfinite(trn_tst))[0]
            trn = np.empty_like(trn_tst)
            trn[:] = np.nan
            tst = np.empty_like(trn_tst)
            tst[:] = np.nan
            trn[unq_col[last_tst_col]:] = trn_tst[unq_col[last_tst_col]:]
            tst[:unq_col[last_tst_col]] = trn_tst[:unq_col[last_tst_col]]

        pdb.set_trace()
        np.random.seed(seed=randomSeed)
        if sparseCustom is not None:
            n_profiles = np.nonzero(np.count_nonzero(~np.isnan(trn), axis=0))[0].shape[0] #n nonzero columns
            n_profiles_to_zero = n_profiles - sparseCustom #n nonzero columns
            if n_profiles_to_zero < 0:
                print("not enough training obs")
                return((sparseCustom,None,None,None,None,None,None,None,None))
            profiles_ind = np.nonzero(np.count_nonzero(~np.isnan(trn), axis=0))[0] #nonzero columns
            index = np.random.choice(profiles_ind.shape[0], n_profiles_to_zero, replace=False)  
            trn[:,profiles_ind[index]] = np.nan
            # tst = trn
        #convert dates to numpy datetime64
        # dates = [date.decode() for date in dates]
        dates = pd.to_datetime(dates, format='%Y-%m-%d')
        dates = np.array(dates,dtype=np.datetime64)

        years = dates.astype('datetime64[Y]').astype(int) + 1970
        assert np.isfinite(feat_mat).all(), "feat_mat has nan at" + str(np.argwhere(np.isfinite(feat_mat)))
        assert np.isfinite(feat_mat_raw).all(), "feat_mat_raw has nan at" + str(np.argwhere(np.isfinite(feat_mat_raw)))
        # assert np.isfinite(Y_mat).any(), "Y_mat has nan at" + str(np.argwhere(np.isfinite(Y_mat)))

        n_depths = 1
        assert feat_mat.shape[0] == feat_mat_raw.shape[0]
        assert feat_mat.shape[0] == tst.shape[0]
        assert feat_mat.shape[0] == trn.shape[0]
        win_shift_tst = begin_loss_ind
        udates = dates
        n_dates = feat_mat.shape[0]
        if verbose:
            print("n dates: ", n_dates)
        seq_per_depth = math.floor(n_dates / seq_length)
        train_seq_per_depth = seq_per_depth
        test_seq_per_depth = seq_per_depth
        win_per_seq = math.floor(seq_length / win_shift) - 1 #windows per sequence (only training)
        win_per_seq = 2#windows per sequence (only training)
        tst_win_per_seq = 2 #windows per sequence (only training)
        n_train_seq = train_seq_per_depth * win_per_seq
        if n_dates % seq_length > 0 and n_dates - seq_length > 0:
            n_train_seq += 1

        if debug:
            print("n train seq: ", n_train_seq)

        n_train_seq_no_window = train_seq_per_depth 
        if lakename == '121623043':
            pdb.set_trace()
        last_test_date_ind = np.where(np.isfinite(tst))[0][-1]
        n_test_seq = (test_seq_per_depth) * tst_win_per_seq
        if last_test_date_ind % seq_length > 0 and last_test_date_ind - seq_length > 0:
            n_test_seq += 1


        n_all_seq = n_train_seq_no_window 


        #build train and test sets, add all data for physical loss

        X_trn = np.empty(shape=(n_train_seq, seq_length, n_features+1)) #features + label
        X_tst = np.empty(shape=(n_test_seq, seq_length, n_features+1))
        trn_dates = np.empty(shape=(n_train_seq, seq_length), dtype='datetime64[s]')
        tst_dates = np.empty(shape=(n_test_seq, seq_length), dtype='datetime64[s]')
        trn_dates[:] = np.datetime64("NaT")
        tst_dates[:] = np.datetime64("NaT")

        X_all = np.empty(shape=(n_all_seq, seq_length, n_features+1))
        all_dates = np.empty(shape=(n_all_seq, seq_length), dtype='datetime64[s]')
        X_all = np.empty(shape=(n_all_seq, seq_length, n_features+1))
        # X_phys = np.empty(shape=(n_all_seq, seq_length, n_features+1)) #non-normalized features + ice cover flag
        X_phys = np.empty(shape=(n_all_seq, seq_length, n_features)) #non-normalized features + ice cover flag

        X_trn[:] = np.nan
        X_tst[:] = np.nan
        X_all[:] = np.nan
        X_phys[:] = np.nan

        #seq index for data to be returned
        tr_seq_ind = 0 
        ts_seq_ind = 0
        all_seq_ind = 0
        #build datasets
        del_all_seq = 0
        if debug:
            print("x_trn shape prior to populating ", X_trn.shape)
        # print("obs index: ", np.where(np.isfinite(trn)))
        for s in range(0,train_seq_per_depth):
            start_index = s*seq_length
            end_index = (s+1)*seq_length
            if end_index > n_dates:
                n_train_seq -= win_per_seq*n_depths
                n_all_seq -= n_depths
                del_all_seq += 1
                X_all = np.delete(X_all, np.arange(X_all.shape[0],X_all.shape[0]-n_depths,-1), axis=0)
                X_trn = np.delete(X_trn, np.arange(X_trn.shape[0],X_trn.shape[0]-win_per_seq*n_depths,-1), axis=0)
                trn_dates = np.delete(trn_dates, np.arange(trn_dates.shape[0], trn_dates.shape[0] - n_depths*win_per_seq,-1), axis=0)
                continue
            X_all[all_seq_ind, :, :-1] = feat_mat[start_index:end_index,:] #feat
            all_dates[all_seq_ind, :] = dates[start_index:end_index] #dates
            X_all[all_seq_ind,:,-1] = np.nan #no label
            X_phys[all_seq_ind, :, :] = feat_mat_raw[start_index:end_index,:]
            all_seq_ind += 1   

            #now do sliding windows for training data 
            for w in range(0, win_per_seq):
                win_start_ind = start_index + w*win_shift
                win_end_ind = win_start_ind + seq_length
                if win_end_ind > n_dates:
                    n_train_seq -= 1
                    X_trn = np.delete(X_trn, -1, axis=0)
                    trn_dates = np.delete(trn_dates, -1, axis=0)
                    continue
                X_trn[tr_seq_ind, :, :-1] = feat_mat[win_start_ind:win_end_ind,:]
                X_trn[tr_seq_ind,:,-1] = trn[win_start_ind:win_end_ind]
                trn_dates[tr_seq_ind,:] = dates[win_start_ind:win_end_ind]
                tr_seq_ind += 1
        #final seq starts at end and goes inward [seq_length]
        # print("n dates: ", n_dates, ", seq len: ", seq_length)
        if n_dates % seq_length > 0:
            end_ind = n_dates
            start_ind = end_ind - seq_length
            X_trn[tr_seq_ind, :, :-1] = feat_mat[start_ind:end_ind,:]
            X_trn[tr_seq_ind,:,-1] = trn[start_ind:end_ind]
            trn_dates[tr_seq_ind,:] = dates[start_ind:end_ind]
            tr_seq_ind += 1

        if debug:
            print("x_trn shape after populating ", X_trn.shape)
        #assert data was constructed correctly
        if tr_seq_ind != n_train_seq:
            # print("incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind))
            extra = n_train_seq - tr_seq_ind
            n_train_seq -= extra
            X_trn = np.delete(X_trn, np.arange(X_trn.shape[0],X_trn.shape[0]-extra,-1), axis=0)
            trn_dates = np.delete(trn_dates, np.arange(X_trn.shape[0],X_trn.shape[0]-extra,-1), axis=0)
        assert tr_seq_ind == n_train_seq, \
         "incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind)
        if debug:
            print("x_trn shape after removing extra seqs", X_trn.shape)
        while trn_dates[-1,0] == np.datetime64("NaT"):
            trn_dates = np.delete(trn_dates, -1, axis=0)
            X_trn = np.delete(X_trn, -1, axis=0)
        if debug:
            print("x_trn shape after removing NaT", X_trn.shape)
        if n_test_seq != 0:
            #now test data(maybe bug in this specification of end of range?)
            for s in range(test_seq_per_depth):
                    start_index = s*seq_length
                    end_index = (s+1)*seq_length
                    if end_index > n_dates:
                        n_test_seq -= tst_win_per_seq
                        X_tst = np.delete(X_tst, np.arange(X_tst.shape[0], X_tst.shape[0] - tst_win_per_seq,-1), axis=0)
                        tst_dates = np.delete(tst_dates, np.arange(tst_dates.shape[0], tst_dates.shape[0] - tst_win_per_seq,-1), axis=0)
                        continue
                    for w in range(0, tst_win_per_seq):
                        win_start_ind = start_index+w*win_shift_tst
                        win_end_ind = win_start_ind + seq_length
                        if win_end_ind > n_dates:
                            n_test_seq -= 1
                            X_tst = np.delete(X_tst, np.arange(X_tst.shape[0], X_tst.shape[0] - 1,-1), axis=0)
                            tst_dates = np.delete(tst_dates, np.arange(tst_dates.shape[0], tst_dates.shape[0] - 1,-1), axis=0)
                            continue

                        X_tst[ts_seq_ind, :, :-1] = feat_mat[win_start_ind:win_end_ind,:]
                        X_tst[ts_seq_ind,:,-1] = tst[win_start_ind:win_end_ind]
                        tst_dates[ts_seq_ind,:] = dates[win_start_ind:win_end_ind]
                        ts_seq_ind += 1
                               #final seq starts at end and goes inward [seq_length]
            if debug:
                print("last_test_date_ind: ", last_test_date_ind, ", sl ", seq_length)

            if last_test_date_ind % seq_length > 0 and last_test_date_ind - seq_length > 0:
                end_ind = last_test_date_ind
                if not end_ind - seq_length < 0:
                    start_ind = end_ind - seq_length

                    X_tst[ts_seq_ind, :, :-1] = feat_mat[start_ind:end_ind,:]
                    X_tst[ts_seq_ind,:,-1] = trn[start_ind:end_ind]
                    tst_dates[ts_seq_ind,:] = dates[start_ind:end_ind]
                    ts_seq_ind += 1
        while np.isnat(tst_dates[-1,0]):
            if debug:
                print("NaT?")
            tst_dates = np.delete(tst_dates, -1, axis=0)
            X_tst = np.delete(X_tst, -1, axis=0)




        #assert data was constructed correctly
        assert ts_seq_ind == n_test_seq, \
            "incorrect number of tst seq estimated {} vs actual{}".format(n_test_seq, ts_seq_ind)      
        #remove sequences with no labels
        tr_seq_removed = 0
        trn_del_ind = np.array([], dtype=np.int32)
        ts_seq_removed = 0
        tst_del_ind = np.array([], dtype=np.int32)

        #if allTestSeq, combine trn and test unfiltered
        if allTestSeq:
            X_tst = np.vstack((X_tst, X_trn))
            tst_dates = np.vstack((tst_dates, trn_dates))
            while np.isnat(tst_dates[-1,0]):
                tst_dates = np.delete(tst_dates, -1, axis=0)
                X_tst = np.delete(X_tst, -1, axis=0)
        # full_data = 
        for i in range(X_trn.shape[0]):

            # print("seq ",i," nz-val-count:",np.count_nonzero(~np.isnan(X_trn[i,:,-1])))
            print("seq ",i)
            if not np.isfinite(X_trn[i,:,:-1]).all():
                print("MISSING FEAT REMOVE")
                tr_seq_removed += 1
                trn_del_ind = np.append(trn_del_ind, i)
            if np.isfinite(X_trn[i,begin_loss_ind:,-1]).any():
                print("HAS OBSERVE, CONTINUE")
                continue
            else:
                # print(X_trn[i,:,-1])
                print("NO OBSERVE, REMOVE")
                tr_seq_removed += 1
                trn_del_ind = np.append(trn_del_ind, i)
        if not allTestSeq: #if we don't want to output ALL the data as test
            for i in range(X_tst.shape[0]):
                if not np.isfinite(X_tst[i,:,:-1]).all():
                    ts_seq_removed += 1
                    tst_del_ind = np.append(tst_del_ind, i)
                if np.isfinite(X_tst[i,begin_loss_ind:,-1]).any():
                    continue
                else:
                    tst_del_ind = np.append(tst_del_ind, i)
                    ts_seq_removed += 1
                if i > 0:
                    if tst_dates[i-1,0] > tst_dates[i,0]:
                        if debug:
                            print("date thing?")
                        tst_del_ind = np.append(tst_del_ind, i)
        else:
            for i in range(X_tst.shape[0]):
                if not np.isfinite(X_tst[i,:,:-1]).all():
                    ts_seq_removed += 1
                    tst_del_ind = np.append(tst_del_ind, i)
        #remove denoted values from trn and tst
        X_trn_tmp = np.delete(X_trn, trn_del_ind, axis=0)
        trn_dates_tmp = np.delete(trn_dates, trn_del_ind, axis=0)
        X_tst_tmp = np.delete(X_tst, tst_del_ind, axis=0)
        tst_dates_tmp = np.delete(tst_dates, tst_del_ind, axis=0)
        X_trn = X_trn_tmp
        trn_dates = trn_dates_tmp
        X_tst = X_tst_tmp
        tst_dates = tst_dates_tmp
        #gather unique test dates
        tst_date_lower_bound = np.where(dates == tst_dates[0][0])[0][0]
        tst_date_upper_bound = np.where(dates == tst_dates[-1][-1])[0][0]

        unique_tst_dates = dates[tst_date_lower_bound:tst_date_upper_bound]




        assert np.isfinite(X_trn[:,:,:-1]).all(), "X_trn has nan"
        # assert np.isfinite(all_dates).any(), "all_dates has nan"
        X_trn_comp = torch.cat([X_trn_comp,torch.from_numpy(X_trn).float()],dim=0)
        trn_dates_comp = np.vstack([trn_dates_comp,trn_dates])
        X_tst_comp = torch.cat([X_tst_comp,torch.from_numpy(X_tst).float()],dim=0)
        tst_dates_comp = np.vstack([tst_dates_comp,tst_dates])


        assert np.isfinite(X_all[:,:,:-1]).all(), "X_all has nan"
        assert np.isfinite(X_trn[:,:,:-1]).all(), "X_trn has nan"
        assert np.isfinite(X_phys).all(), "X_phys has nan"
        # assert np.isfinite(all_dates).any(), "all_dates has nan"
        return (X_trn_comp, trn_dates_comp, X_tst_comp, tst_dates_comp, unique_tst_dates)

def buildLakeDataForRNN_manylakes_finetune2(lakename, data_dir, seq_length, n_features, \
                                            win_shift= 1, begin_loss_ind = 100, \
                                            outputFullTestMatrix=False, sparseCustom=None, \
                                            allTestSeq=False, static_feats=False,n_static_feats=0,\
                                            postProcessSplits=True, randomSeed=0):

    #NONAN
    #PARAMETERS
        #@lakename = string of lake name as the folder of /data/processed/{lakename}
        #@seq_length = sequence length of LSTM inputs
        #@n_features = number of physical drivers
        #@win_shift = days to move in the sliding window for the training set
        #@begin_loss_ind = index in sequence to begin calculating loss function (to avoid poor accuracy in early parts of the sequence)
    #load data created in preprocess.py based on lakename
    n_features = n_features+n_static_feats
    debug = False
    verbose = False
    my_path = os.path.abspath(os.path.dirname(__file__))


    feat_mat_raw = None
    feat_mat = None
    if static_feats:
        feat_mat_raw = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/features_ea.npy"))
        feat_mat = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/processed_features_ea.npy"))
    else:
        feat_mat_raw = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/features.npy"))
        feat_mat = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/processed_features.npy"))

    tst = []
    trn = []

    #GET TRAIN/TEST HERE
    #alternative way to divide test/train just by 1/3rd 2/3rd
    if os.path.exists(os.path.join(my_path, "../../data/processed/"+lakename+"/full.npy")):
        full = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/full.npy"))
    else:
        full = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/full_obs.npy"))


    dates = []
    if os.path.exists(os.path.join(my_path, "../../data/processed/"+lakename+"/dates.npy")):
        dates = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/dates.npy"))
    else:
        dates = []

    #post process train/test splits
    if postProcessSplits:
        # shape0 = trn.shape[0]
        # trn_flt = trn.flatten()
        # tst_flt = tst.flatten()
        # np.put(trn_flt, np.where(np.isfinite(tst_flt))[0], tst_flt[np.isfinite(tst_flt)])
        trn_tst = full.flatten()
        # trn_tst = trn_flt.reshape((shape0))
        last_tst_col = int(np.round(np.unique(np.where(np.isfinite(trn_tst))).shape[0]/3))
        unq_col = np.unique(np.where(np.isfinite(trn_tst))[1])
        trn = np.empty_like(trn_tst)
        trn[:] = np.nan
        tst = np.empty_like(trn_tst)
        tst[:] = np.nan
        trn[:,unq_col[last_tst_col]:] = trn_tst[:,unq_col[last_tst_col]:]
        tst[:,:unq_col[last_tst_col]] = trn_tst[:,:unq_col[last_tst_col]]

    np.random.seed(seed=randomSeed)
    if sparseCustom is not None:
        n_profiles = np.nonzero(np.count_nonzero(~np.isnan(trn), axis=0))[0].shape[0] #n nonzero columns
        n_profiles_to_zero = n_profiles - sparseCustom #n nonzero columns
        if n_profiles_to_zero < 0:
            print("not enough training obs")
            return((sparseCustom,None,None,None,None,None,None,None,None))
        profiles_ind = np.nonzero(np.count_nonzero(~np.isnan(trn), axis=0))[0] #nonzero columns
        index = np.random.choice(profiles_ind.shape[0], n_profiles_to_zero, replace=False)  
        trn[:,profiles_ind[index]] = np.nan
        # tst = trn
    #convert dates to numpy datetime64
    # dates = [date.decode() for date in dates]
    dates = pd.to_datetime(dates, format='%Y-%m-%d')
    dates = np.array(dates,dtype=np.datetime64)

    years = dates.astype('datetime64[Y]').astype(int) + 1970
    assert np.isfinite(feat_mat).all(), "feat_mat has nan at" + str(np.argwhere(np.isfinite(feat_mat)))
    assert np.isfinite(feat_mat_raw).all(), "feat_mat_raw has nan at" + str(np.argwhere(np.isfinite(feat_mat_raw)))
    # assert np.isfinite(Y_mat).any(), "Y_mat has nan at" + str(np.argwhere(np.isfinite(Y_mat)))

    n_depths = feat_mat.shape[0]
    assert feat_mat.shape[0] == feat_mat_raw.shape[0]
    assert feat_mat.shape[0] == tst.shape[0]
    assert feat_mat.shape[0] == trn.shape[0]
    win_shift_tst = begin_loss_ind
    udates = dates
    n_dates = feat_mat.shape[0]
    if verbose:
        print("n dates: ", n_dates)
    seq_per_depth = math.floor(n_dates / seq_length)
    train_seq_per_depth = seq_per_depth
    test_seq_per_depth = seq_per_depth
    win_per_seq = math.floor(seq_length / win_shift) - 1 #windows per sequence (only training)
    win_per_seq = 2#windows per sequence (only training)
    tst_win_per_seq = 2 #windows per sequence (only training)
    n_train_seq = train_seq_per_depth * win_per_seq
    if n_dates % seq_length > 0 and n_dates - seq_length > 0:
        n_train_seq += 1

    if debug:
        print("n train seq: ", n_train_seq)

    n_train_seq_no_window = train_seq_per_depth 
    if lakename == '121623043':
        pdb.set_trace()
    last_test_date_ind = np.where(np.isfinite(tst))[0][-1]
    n_test_seq = (test_seq_per_depth) * tst_win_per_seq
    if last_test_date_ind % seq_length > 0 and last_test_date_ind - seq_length > 0:
        n_test_seq += 1


    n_all_seq = n_train_seq_no_window 


    #build train and test sets, add all data for physical loss

    X_trn = np.empty(shape=(n_train_seq, seq_length, n_features+1)) #features + label
    X_tst = np.empty(shape=(n_test_seq, seq_length, n_features+1))
    trn_dates = np.empty(shape=(n_train_seq, seq_length), dtype='datetime64[s]')
    tst_dates = np.empty(shape=(n_test_seq, seq_length), dtype='datetime64[s]')
    trn_dates[:] = np.datetime64("NaT")
    tst_dates[:] = np.datetime64("NaT")

    X_all = np.empty(shape=(n_all_seq, seq_length, n_features+1))
    all_dates = np.empty(shape=(n_all_seq, seq_length), dtype='datetime64[s]')
    X_all = np.empty(shape=(n_all_seq, seq_length, n_features+1))
    X_phys = np.empty(shape=(n_all_seq, seq_length, n_features+1)) #non-normalized features + ice cover flag

    X_trn[:] = np.nan
    X_tst[:] = np.nan
    X_all[:] = np.nan
    X_phys[:] = np.nan

    #seq index for data to be returned
    tr_seq_ind = 0 
    ts_seq_ind = 0
    all_seq_ind = 0
    #build datasets
    del_all_seq = 0
    if debug:
        print("x_trn shape prior to populating ", X_trn.shape)
    # print("obs index: ", np.where(np.isfinite(trn)))
    for s in range(0,train_seq_per_depth):
        start_index = s*seq_length
        end_index = (s+1)*seq_length
        if end_index > n_dates:
            n_train_seq -= win_per_seq*n_depths
            n_all_seq -= n_depths
            del_all_seq += 1
            X_all = np.delete(X_all, np.arange(X_all.shape[0],X_all.shape[0]-n_depths,-1), axis=0)
            X_trn = np.delete(X_trn, np.arange(X_trn.shape[0],X_trn.shape[0]-win_per_seq*n_depths,-1), axis=0)
            trn_dates = np.delete(trn_dates, np.arange(trn_dates.shape[0], trn_dates.shape[0] - n_depths*win_per_seq,-1), axis=0)
            continue
        X_all[all_seq_ind, :, :-1] = feat_mat[start_index:end_index,:] #feat
        all_dates[all_seq_ind, :] = dates[start_index:end_index] #dates
        X_all[all_seq_ind,:,-1] = np.nan #no label
        X_phys[all_seq_ind, :, :] = feat_mat_raw[start_index:end_index,:]
        all_seq_ind += 1   

        #now do sliding windows for training data 
        for w in range(0, win_per_seq):
            win_start_ind = start_index + w*win_shift
            win_end_ind = win_start_ind + seq_length


            if win_end_ind > n_dates:
                n_train_seq -= 1
                X_trn = np.delete(X_trn, -1, axis=0)
                trn_dates = np.delete(trn_dates, -1, axis=0)
                continue
            X_trn[tr_seq_ind, :, :-1] = feat_mat[win_start_ind:win_end_ind,:]
            X_trn[tr_seq_ind,:,-1] = trn[win_start_ind:win_end_ind]
            trn_dates[tr_seq_ind,:] = dates[win_start_ind:win_end_ind]
            tr_seq_ind += 1
    #final seq starts at end and goes inward [seq_length]
    # print("n dates: ", n_dates, ", seq len: ", seq_length)
    if n_dates % seq_length > 0:
        end_ind = n_dates
        start_ind = end_ind - seq_length
        X_trn[tr_seq_ind, :, :-1] = feat_mat[start_ind:end_ind,:]
        X_trn[tr_seq_ind,:,-1] = trn[start_ind:end_ind]
        trn_dates[tr_seq_ind,:] = dates[start_ind:end_ind]
        tr_seq_ind += 1

    if debug:
        print("x_trn shape after populating ", X_trn.shape)
    #assert data was constructed correctly
    if tr_seq_ind != n_train_seq:
        # print("incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind))
        extra = n_train_seq - tr_seq_ind
        n_train_seq -= extra
        X_trn = np.delete(X_trn, np.arange(X_trn.shape[0],X_trn.shape[0]-extra,-1), axis=0)
        trn_dates = np.delete(trn_dates, np.arange(X_trn.shape[0],X_trn.shape[0]-extra,-1), axis=0)
    assert tr_seq_ind == n_train_seq, \
     "incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind)

    while trn_dates[-1,0] == np.datetime64("NaT"):
        trn_dates = np.delete(trn_dates, -1, axis=0)
        X_trn = np.delete(X_trn, -1, axis=0)

    if n_test_seq != 0:
        #now test data(maybe bug in this specification of end of range?)
        for s in range(test_seq_per_depth):
                start_index = s*seq_length
                end_index = (s+1)*seq_length
                if end_index > n_dates:
                    n_test_seq -= tst_win_per_seq
                    X_tst = np.delete(X_tst, np.arange(X_tst.shape[0], X_tst.shape[0] - tst_win_per_seq,-1), axis=0)
                    tst_dates = np.delete(tst_dates, np.arange(tst_dates.shape[0], tst_dates.shape[0] - tst_win_per_seq,-1), axis=0)
                    continue
                for w in range(0, tst_win_per_seq):
                    win_start_ind = start_index+w*win_shift_tst
                    win_end_ind = win_start_ind + seq_length
                    if win_end_ind > n_dates:
                        n_test_seq -= 1
                        X_tst = np.delete(X_tst, np.arange(X_tst.shape[0], X_tst.shape[0] - 1,-1), axis=0)
                        tst_dates = np.delete(tst_dates, np.arange(tst_dates.shape[0], tst_dates.shape[0] - 1,-1), axis=0)
                        continue

                    X_tst[ts_seq_ind, :, :-1] = feat_mat[win_start_ind:win_end_ind,:]
                    X_tst[ts_seq_ind,:,-1] = tst[win_start_ind:win_end_ind]
                    tst_dates[ts_seq_ind,:] = dates[win_start_ind:win_end_ind]
                    ts_seq_ind += 1
                           #final seq starts at end and goes inward [seq_length]
        if debug:
            print("last_test_date_ind: ", last_test_date_ind, ", sl ", seq_length)

        if last_test_date_ind % seq_length > 0 and last_test_date_ind - seq_length > 0:
            end_ind = last_test_date_ind
            if not end_ind - seq_length < 0:
                start_ind = end_ind - seq_length

                X_tst[ts_seq_ind, :, :-1] = feat_mat[start_ind:end_ind,:]
                X_tst[ts_seq_ind,:,-1] = trn[start_ind:end_ind]
                tst_dates[ts_seq_ind,:] = dates[start_ind:end_ind]
                ts_seq_ind += 1
    while np.isnat(tst_dates[-1,0]):
        if debug:
            print("NaT?")
        tst_dates = np.delete(tst_dates, -1, axis=0)
        X_tst = np.delete(X_tst, -1, axis=0)




    #assert data was constructed correctly
    assert ts_seq_ind == n_test_seq, \
        "incorrect number of tst seq estimated {} vs actual{}".format(n_test_seq, ts_seq_ind)      
    #remove sequences with no labels
    tr_seq_removed = 0
    trn_del_ind = np.array([], dtype=np.int32)
    ts_seq_removed = 0
    tst_del_ind = np.array([], dtype=np.int32)

    #if allTestSeq, combine trn and test unfiltered
    if allTestSeq:
        X_tst = np.vstack((X_tst, X_trn))
        tst_dates = np.vstack((tst_dates, trn_dates))
        while np.isnat(tst_dates[-1,0]):
            tst_dates = np.delete(tst_dates, -1, axis=0)
            X_tst = np.delete(X_tst, -1, axis=0)
    # full_data = 
    for i in range(X_trn.shape[0]):
        # print("seq ",i," nz-val-count:",np.count_nonzero(~np.isnan(X_trn[i,:,-1])))
        if not np.isfinite(X_trn[i,:,:-1]).all():
            # print("MISSING FEAT REMOVE")
            tr_seq_removed += 1
            trn_del_ind = np.append(trn_del_ind, i)
        if np.isfinite(X_trn[i,begin_loss_ind:,-1]).any():
            # print("HAS OBSERVE, CONTINUE")
            continue
        else:
            # print(X_trn[i,:,-1])
            # print("NO OBSERVE, REMOVE")
            tr_seq_removed += 1
            trn_del_ind = np.append(trn_del_ind, i)
    if not allTestSeq: #if we don't want to output ALL the data as test
        for i in range(X_tst.shape[0]):
            if not np.isfinite(X_tst[i,:,:-1]).all():
                ts_seq_removed += 1
                tst_del_ind = np.append(tst_del_ind, i)
            if np.isfinite(X_tst[i,begin_loss_ind:,-1]).any():
                continue
            else:
                tst_del_ind = np.append(tst_del_ind, i)
                ts_seq_removed += 1
            if i > 0:
                if tst_dates[i-1,0] > tst_dates[i,0]:
                    if debug:
                        print("date thing?")
                    tst_del_ind = np.append(tst_del_ind, i)
    else:
        for i in range(X_tst.shape[0]):
            if not np.isfinite(X_tst[i,:,:-1]).all():
                ts_seq_removed += 1
                tst_del_ind = np.append(tst_del_ind, i)
    #remove denoted values from trn and tst
    X_trn_tmp = np.delete(X_trn, trn_del_ind, axis=0)
    trn_dates_tmp = np.delete(trn_dates, trn_del_ind, axis=0)
    X_tst_tmp = np.delete(X_tst, tst_del_ind, axis=0)
    tst_dates_tmp = np.delete(tst_dates, tst_del_ind, axis=0)
    X_trn = X_trn_tmp
    trn_dates = trn_dates_tmp
    X_tst = X_tst_tmp
    tst_dates = tst_dates_tmp
    #gather unique test dates
    tst_date_lower_bound = np.where(dates == tst_dates[0][0])[0][0]
    tst_date_upper_bound = np.where(dates == tst_dates[-1][-1])[0][0]

    unique_tst_dates = dates[tst_date_lower_bound:tst_date_upper_bound]


    assert np.isfinite(X_all[:,:,:-1]).all(), "X_all has nan"
    assert np.isfinite(X_trn[:,:,:-1]).all(), "X_trn has nan"
    assert np.isfinite(X_phys).all(), "X_phys has nan"
    # assert np.isfinite(all_dates).any(), "all_dates has nan"
    return (torch.from_numpy(X_trn), trn_dates, torch.from_numpy(X_tst), tst_dates, unique_tst_dates,torch.from_numpy(X_all), 
            torch.from_numpy(X_phys), all_dates)


def buildLakeDataForRNN_conus(lakename, data_dir, seq_length, n_features, \
                                            win_shift= 1, begin_loss_ind = 100, \
                                            outputFullTestMatrix=False, sparseCustom=None, \
                                            allTestSeq=False,n_static_feats=0,removeElevation=False,\
                                            postProcessSplits=True, randomSeed=0):

    #NONAN
    #PARAMETERS
        #@lakename = string of lake name as the folder of /data/processed/{lakename}
        #@seq_length = sequence length of LSTM inputs
        #@n_features = number of physical drivers
        #@win_shift = days to move in the sliding window for the training set
        #@begin_loss_ind = index in sequence to begin calculating loss function (to avoid poor accuracy in early parts of the sequence)
    #load data created in preprocess.py based on lakename
    n_features = n_features+n_static_feats
    debug = False
    verbose = False
    my_path = os.path.abspath(os.path.dirname(__file__))

    feat_mat_raw = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/features_ea_conus_021621.npy"))
    feat_mat = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/processed_features_ea_conus_021621.npy"))
    if removeElevation:
        feat_mat = np.delete(feat_mat,29,axis=1)
        feat_mat_raw = np.delete(feat_mat_raw,29,axis=1)
    if feat_mat.shape[0] < seq_length:
        return (None,None)
    full = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/full.npy"))

    trn = full

    # if debug:
    #     print("initial trn: ", trn)
    #     print("observations: ",np.count_nonzero(~np.isnan(trn)))
    dates = np.load(os.path.join(my_path, "../../data/processed/"+lakename+"/dates.npy"))

    #post process train/test splits

    # if postProcessSplits:
    #     shape0 = trn.shape[0]
    #     trn_flt = trn.flatten()
    #     tst_flt = tst.flatten()
    #     np.put(trn_flt, np.where(np.isfinite(tst_flt))[0], tst_flt[np.isfinite(tst_flt)])
    #     trn_tst = trn_flt.reshape((shape0))
    #     last_tst_col = int(np.round(np.unique(np.where(np.isfinite(trn_tst))).shape[0]/3))
    #     unq_col = np.unique(np.where(np.isfinite(trn_tst))[1])
    #     trn = np.empty_like(trn_tst)
    #     trn[:] = np.nan
    #     tst = np.empty_like(trn_tst)
    #     tst[:] = np.nan
    #     trn[:,unq_col[last_tst_col]:] = trn_tst[:,unq_col[last_tst_col]:]
    #     tst[:,:unq_col[last_tst_col]] = trn_tst[:,:unq_col[last_tst_col]]

    np.random.seed(seed=randomSeed)
    if sparseCustom is not None:
        n_profiles = np.nonzero(np.count_nonzero(~np.isnan(trn), axis=0))[0].shape[0] #n nonzero columns
        n_profiles_to_zero = n_profiles - sparseCustom #n nonzero columns
        if n_profiles_to_zero < 0:
            print("not enough training obs")
            return((sparseCustom,None,None,None,None,None,None,None,None))
        profiles_ind = np.nonzero(np.count_nonzero(~np.isnan(trn), axis=0))[0] #nonzero columns
        index = np.random.choice(profiles_ind.shape[0], n_profiles_to_zero, replace=False)  
        trn[:,profiles_ind[index]] = np.nan
        # tst = trn
    #convert dates to numpy datetime64
    # dates = [date.decode() for date in dates]
    dates = pd.to_datetime(dates, format='%Y-%m-%d')
    dates = np.array(dates,dtype=np.datetime64)

    years = dates.astype('datetime64[Y]').astype(int) + 1970
    assert np.isfinite(feat_mat).all(), "feat_mat has nan at" + str(np.argwhere(np.isfinite(feat_mat)))
    assert np.isfinite(feat_mat_raw).all(), "feat_mat_raw has nan at" + str(np.argwhere(np.isfinite(feat_mat_raw)))
    # assert np.isfinite(Y_mat).any(), "Y_mat has nan at" + str(np.argwhere(np.isfinite(Y_mat)))

    assert feat_mat.shape[0] == feat_mat_raw.shape[0]
    assert feat_mat.shape[0] == trn.shape[0]
    udates = dates
    n_dates = feat_mat.shape[0]
    if verbose:
        print("n dates: ", n_dates)
    seq_per_depth = math.floor(n_dates / seq_length)
    train_seq_per_depth = seq_per_depth
    win_per_seq = math.floor(seq_length / win_shift) - 1 #windows per sequence (only training)
    win_per_seq = 2#windows per sequence (only training)
    tst_win_per_seq = 2 #windows per sequence (only training)
    n_train_seq = train_seq_per_depth * win_per_seq
    if n_dates % seq_length > 0 and n_dates - seq_length > 0:
        n_train_seq += 1

    if debug:
        print("n train seq: ", n_train_seq)

    n_train_seq_no_window = train_seq_per_depth 



    #build train and test sets, add all data for physical loss

    X_trn = np.empty(shape=(n_train_seq, seq_length, n_features+1)) #features + label
    trn_dates = np.empty(shape=(n_train_seq, seq_length), dtype='datetime64[s]')

    X_trn[:] = np.nan

    #seq index for data to be returned
    tr_seq_ind = 0 
    #build datasets
    del_all_seq = 0
    if debug:
        print("x_trn shape prior to populating ", X_trn.shape)
    # print("obs index: ", np.where(np.isfinite(trn)))
    for s in range(0,train_seq_per_depth):
        start_index = s*seq_length
        end_index = (s+1)*seq_length
        if end_index > n_dates:
            n_train_seq -= win_per_seq*n_depths
            n_all_seq -= n_depths
            del_all_seq += 1
            X_all = np.delete(X_all, np.arange(X_all.shape[0],X_all.shape[0]-n_depths,-1), axis=0)
            X_trn = np.delete(X_trn, np.arange(X_trn.shape[0],X_trn.shape[0]-win_per_seq*n_depths,-1), axis=0)
            trn_dates = np.delete(trn_dates, np.arange(trn_dates.shape[0], trn_dates.shape[0] - n_depths*win_per_seq,-1), axis=0)
            continue

        #now do sliding windows for training data 
        for w in range(0, win_per_seq):
            win_start_ind = start_index + w*win_shift
            win_end_ind = win_start_ind + seq_length


            if win_end_ind > n_dates:
                n_train_seq -= 1
                X_trn = np.delete(X_trn, -1, axis=0)
                trn_dates = np.delete(trn_dates, -1, axis=0)
                continue
            X_trn[tr_seq_ind, :, :-1] = feat_mat[win_start_ind:win_end_ind,:]
            X_trn[tr_seq_ind,:,-1] = trn[win_start_ind:win_end_ind]
            trn_dates[tr_seq_ind,:] = dates[win_start_ind:win_end_ind]
            tr_seq_ind += 1
    #final seq starts at end and goes inward [seq_length]
    # print("n dates: ", n_dates, ", seq len: ", seq_length)
    if n_dates % seq_length > 0:
        end_ind = n_dates
        start_ind = end_ind - seq_length
        # if lakename == "nhdhr_57937006":
        #     pdb.set_trace()
        X_trn[tr_seq_ind, :, :-1] = feat_mat[start_ind:end_ind,:]
        X_trn[tr_seq_ind,:,-1] = trn[start_ind:end_ind]
        trn_dates[tr_seq_ind,:] = dates[start_ind:end_ind]
        tr_seq_ind += 1

    if debug:
        print("x_trn shape after populating ", X_trn.shape)
    #assert data was constructed correctly
    if tr_seq_ind != n_train_seq:
        # print("incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind))
        extra = n_train_seq - tr_seq_ind
        n_train_seq -= extra
        X_trn = np.delete(X_trn, np.arange(X_trn.shape[0],X_trn.shape[0]-extra,-1), axis=0)
        trn_dates = np.delete(trn_dates, np.arange(X_trn.shape[0],X_trn.shape[0]-extra,-1), axis=0)
    assert tr_seq_ind == n_train_seq, \
     "incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind)

    while trn_dates[-1,0] == np.datetime64("NaT"):
        trn_dates = np.delete(trn_dates, -1, axis=0)
        X_trn = np.delete(X_trn, -1, axis=0)

 




    #remove sequences with no labels
    tr_seq_removed = 0
    trn_del_ind = np.array([], dtype=np.int32)

    #if allTestSeq, combine trn and test unfiltered
    for i in range(X_trn.shape[0]):
        # print("seq ",i," nz-val-count:",np.count_nonzero(~np.isnan(X_trn[i,:,-1])))
        if not np.isfinite(X_trn[i,:,:-1]).all():
            # print("MISSING FEAT REMOVE")
            tr_seq_removed += 1
            trn_del_ind = np.append(trn_del_ind, i)
        if np.isfinite(X_trn[i,begin_loss_ind:,-1]).any():
            # print("HAS OBSERVE, CONTINUE")
            continue
        else:
            # print(X_trn[i,:,-1])
            # print("NO OBSERVE, REMOVE")
            tr_seq_removed += 1
            trn_del_ind = np.append(trn_del_ind, i)

    #remove denoted values from trn and tst
    X_trn_tmp = np.delete(X_trn, trn_del_ind, axis=0)
    trn_dates_tmp = np.delete(trn_dates, trn_del_ind, axis=0)
    X_trn = X_trn_tmp
    trn_dates = trn_dates_tmp
    #gather unique test dates


    assert np.isfinite(X_trn[:,:,:-1]).all(), "X_trn has nan"
    # assert np.isfinite(all_dates).any(), "all_dates has nan"
    return (torch.from_numpy(X_trn), trn_dates)




def buildLakeDataForRNNPretrain(lakename, data_dir, seq_length, n_features, win_shift= 1, begin_loss_ind = 102, excludeTest=False, targetLake = None, normAll=False, normGE10=False):
    #PARAMETERS
        #@lakename = string of lake name as the folder of /data/processed/{lakename}
        #@seq_length = sequence length of LSTM inputs
        #@n_features = number of physical drivers
        #@win_shift = days to move in the sliding window for the training set
    #load data created in preprocess.py based on lakename
    my_path = os.path.abspath(os.path.dirname(__file__))

    feat_mat_raw = np.load(os.path.join(my_path, data_dir +"features_pt.npy"))
    feat_mat = np.load(os.path.join(my_path, data_dir + "processed_features_pt.npy"))
    Y_mat = np.load(os.path.join(my_path, data_dir +"glm_pt.npy"))

    dates = np.load(os.path.join(my_path, data_dir +"dates_pt.npy"))

    years = dates.astype('datetime64[Y]').astype(int) + 1970
 
    #for debugging
    debug = False

    assert np.isfinite(feat_mat).all(), "feat_mat has nan at" + str(np.argwhere(np.isfinite(feat_mat)))
    assert np.isfinite(feat_mat_raw).all(), "feat_mat_raw has nan at" + str(np.argwhere(np.isfinite(feat_mat_raw)))
    assert np.isfinite(Y_mat).all(), "Y_mat has nan at" + str(np.argwhere(np.isfinite(Y_mat)))

    assert feat_mat.shape[0] == feat_mat_raw.shape[0]
    assert feat_mat.shape[0] == Y_mat.shape[0]
    udates = dates
    n_dates = feat_mat.shape[0]
    seq_per_depth = math.floor(n_dates / seq_length)
    train_seq_per_depth = seq_per_depth
    win_per_seq = math.floor(seq_length / win_shift)  #windows per sequence (only training)
    n_train_seq = train_seq_per_depth * win_per_seq
    n_train_seq_no_window = train_seq_per_depth
    n_all_seq = n_train_seq_no_window

    if debug:
        print("sequences: ", seq_per_depth)
        print("window per seq: ", win_per_seq)

    #build train and test sets, add all data for physical loss
    X_trn = np.empty(shape=(n_train_seq, seq_length, n_features+1)) #features + label

    X_all = np.empty(shape=(n_all_seq, seq_length, n_features+1))
    all_dates = np.empty(shape=(n_all_seq, seq_length), dtype='datetime64[s]')
    X_phys = np.empty(shape=(n_all_seq, seq_length, n_features+1)) #meteo plus, ice

    X_trn[:] = np.nan
    X_all[:] = np.nan
    X_phys[:] = np.nan

    #seq index for data to be returned
    tr_seq_ind = 0 
    all_seq_ind = 0
    s_skipped = 0
    #build datasets
    for s in range(0,train_seq_per_depth):
        start_index = s*seq_length
        end_index = (s+1)*seq_length
        #first do total model data
        X_all[all_seq_ind, :, :-1] = feat_mat[start_index:end_index,:] #feat
        all_dates[all_seq_ind, :] = dates[start_index:end_index] #dates
        X_all[all_seq_ind,:,-1] = Y_mat[start_index:end_index] #label
        X_phys[all_seq_ind, :, :] = feat_mat_raw[start_index:end_index,:] #exclude ice flag
        all_seq_ind += 1   
        #now do sliding windows for training data 
        for w in range(0, win_per_seq):
            win_start_ind = start_index + w*win_shift
            win_end_ind = win_start_ind + seq_length
            if win_end_ind > n_dates:
                continue
            X_trn[tr_seq_ind, :, :-1] = feat_mat[win_start_ind:win_end_ind,:]
            X_trn[tr_seq_ind,:,-1] = Y_mat[win_start_ind:win_end_ind]
            tr_seq_ind += 1
    #assert data was constructed correctly
    if tr_seq_ind != n_train_seq:
        print("incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind))
        extra = n_train_seq - tr_seq_ind
        n_train_seq -= extra
        X_trn = np.delete(X_trn, np.arange(X_trn.shape[0],X_trn.shape[0]-extra,-1), axis=0)
    assert tr_seq_ind == n_train_seq, \
        "incorrect number of trn seq estimated {} vs actual{}".format(n_train_seq, tr_seq_ind)

    # print("train set created")
    

    tr_seq_removed = 0
    trn_del_ind = np.array([], dtype=np.int32)

    for i in range(X_trn.shape[0]):
        # print("seq ",i," nz-val-count:",np.count_nonzero(~np.isnan(X_trn[i,:,-1])))
        if not np.isfinite(X_trn[i,:,:]).all():
            # print("MISSING FEAT REMOVE")
            tr_seq_removed += 1
            trn_del_ind = np.append(trn_del_ind, i)


    #remove denoted values from trn and tst
    X_trn_tmp = np.delete(X_trn, trn_del_ind, axis=0)
    X_trn = X_trn_tmp


    assert np.isfinite(X_trn).all(), "X_trn has nan at" + str(np.argwhere(np.isfinite(X_trn)))
    assert np.isfinite(X_all[:,:,:-1]).all(), "X_all has nan"
    assert np.isfinite(X_phys).any(), "X_phys has nan"
    # assert np.isfinite(all_dates).any(), "all_dates has nan"
    return (torch.from_numpy(X_trn), torch.from_numpy(X_all), torch.from_numpy(X_phys), all_dates)

def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm



def get_energy_diag(inputs, outputs, phys, labels, dates, depth_areas, n_depths, use_gpu, combine_days=1):
    import numpy as np
    n_sets = int(inputs.size()[0] / n_depths) #sets of depths in batch
    diff_vec = torch.empty((inputs.size()[1]))
    n_dates = inputs.size()[1]

    outputs = outputs.view(outputs.size()[0], outputs.size()[1])
    densities = transformTempToDensity(outputs, use_gpu)


    #for experiment
    if use_gpu:
        densities = densities.cuda()  

    #calculate lake energy for each timestep
    lake_energies = calculate_lake_energy(outputs[:,:], densities[:,:], depth_areas)
    #calculate energy change in each timestep
    lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, combine_days, depth_areas[0])
    lake_energy_deltas = lake_energy_deltas[1:]
    lake_energy_fluxes = calculate_energy_fluxes(phys[0,:,:], outputs[0,:], combine_days)
    return (lake_energy_deltas.numpy(), lake_energy_fluxes.numpy())

def calculate_energy(pred, hyps, use_gpu):
    densities = transformTempToDensity(pred, use_gpu)
    if use_gpu:
        densities = densities.cuda() 
    lake_energies = calculate_lake_energy(pred[:,:], densities[:,:], hyps)
    return lake_energies    

def calculate_ec_loss_manylakes(inputs, outputs, phys, labels, dates, depth_areas, n_depths, ec_threshold, use_gpu, combine_days=1):
    import numpy as np
    #******************************************************
    #description: calculates energy conservation loss
    #parameters: 
        #@inputs: features
        #@outputs: labels
        #@phys: features(not standardized) of sw_radiation, lw_radiation, etc
        #@labels modeled temp (will not used in loss, only for test)
        #@depth_areas: cross-sectional area of each depth
        #@n_depths: number of depths
        #@use_gpu: gpu flag
        #@combine_days: how many days to look back to see if energy is conserved
    #*********************************************************************************
    diff_vec = torch.empty((inputs.size()[1]))
    n_dates = inputs.size()[1]
    # outputs = labels  
    outputs = outputs.view(outputs.size()[0], outputs.size()[1])
    # print("modeled temps: ", outputs)
    densities = transformTempToDensity(outputs, use_gpu)
    # print("modeled densities: ", densities)


    #for experiment
    if use_gpu:
        densities = densities.cuda()  
        #loop through sets of n_depths

 

    #calculate lake energy for each timestep

    lake_energies = calculate_lake_energy(outputs[:,:], densities[:,:], depth_areas)
    #calculate energy change in each timestep
    lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, combine_days, depth_areas[0])
    lake_energy_deltas = lake_energy_deltas[1:]
    #calculate sum of energy flux into or out of the lake at each timestep
    # print("dates ", dates[0,1:6])
    lake_energy_fluxes = calculate_energy_fluxes_manylakes(phys[0,:,:], outputs[0,:], combine_days)
    ### can use this to plot energy delta and flux over time to see if they line up
    # doy = np.array([datetime.datetime.combine(date.fromordinal(x), datetime.time.min).timetuple().tm_yday  for x in dates[start_index,:]])
    # doy = doy[1:-1]


    diff_vec = (lake_energy_deltas - lake_energy_fluxes).abs_()
    phys = phys.cpu()
    diff_vec = diff_vec.cpu()
    diff_vec = diff_vec[np.where((phys[0,1:-1,-1] == 0))[0]] #only over no-ice period
    if use_gpu:
        phys = phys.cuda()
        diff_vec = diff_vec.cuda()
    #actual ice
    # diff_vec = diff_vec[np.where((phys[1:(n_depths-diff_vec.size()[0]-1),9] == 0))[0]]
    # #compute difference to be used as penalty
    if diff_vec.size() == torch.Size([0]):
        return 0
    else:
        res = torch.clamp(diff_vec.mean() - ec_threshold, min=0)  
        return res

def calculate_dc_loss(outputs, n_depths, use_gpu):
    #calculates depth-density consistency loss
    #parameters:
        #@outputs: labels = temperature predictions, organized as depth (rows) by date (cols)
        #@n_depths: number of depths
        #@use_gpu: gpu flag

    assert outputs.size()[0] == n_depths

    densities = transformTempToDensity(outputs, use_gpu)

    # We could simply count the number of times that a shallower depth (densities[:-1])
    # has a higher density than the next depth below (densities[1:])
    # num_violations = (densities[:-1] - densities[1:] > 0).sum()

    # But instead, let's use sum(sum(ReLU)) of the density violations,
    # per Karpatne et al. 2018 (https://arxiv.org/pdf/1710.11431.pdf) eq 3.14
    sum_violations = (densities[:-1] - densities[1:]).clamp(min=0).sum()

    return sum_violations


def calculate_ec_loss(inputs, outputs, phys, labels, dates, depth_areas, n_depths, ec_threshold, use_gpu, combine_days=1):
    import numpy as np
    #******************************************************
    #description: calculates energy conservation loss
    #parameters: 
        #@inputs: features
        #@outputs: labels
        #@phys: features(not standardized) of sw_radiation, lw_radiation, etc
        #@labels modeled temp (will not used in loss, only for test)
        #@depth_areas: cross-sectional area of each depth
        #@n_depths: number of depths
        #@use_gpu: gpu flag
        #@combine_days: how many days to look back to see if energy is conserved (obsolete)
    #*********************************************************************************

    n_sets = math.floor(inputs.size()[0] / n_depths)#sets of depths in batch
    diff_vec = torch.empty((inputs.size()[1]))
    n_dates = inputs.size()[1]

    
    outputs = outputs.view(outputs.size()[0], outputs.size()[1])
    # print("modeled temps: ", outputs)
    densities = transformTempToDensity(outputs, use_gpu)
    # print("modeled densities: ", densities)


    #for experiment
    if use_gpu:
        densities = densities.cuda()  
    diff_per_set = torch.empty(n_sets) 
    for i in range(n_sets):
        #loop through sets of n_depths

        #indices
        start_index = (i)*n_depths
        end_index = (i+1)*n_depths


        #assert have all depths
        # assert torch.unique(inputs[:,0,1]).size()[0] == n_depths
        # assert torch.unique(inputs[:,100,1]).size()[0] == n_depths
        # assert torch.unique(inputs[:,200,1]).size()[0] == n_depths
        # assert torch.unique(phys[:,0,1]).size()[0] == n_depths
        # assert torch.unique(phys[:,100,1]).size()[0] == n_depths
        # assert torch.unique(phys[:,200,1]).size()[0] == n_depths


        #calculate lake energy for each timestep
        lake_energies = calculate_lake_energy(outputs[start_index:end_index,:], densities[start_index:end_index,:], depth_areas)
        #calculate energy change in each timestep
        lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, combine_days, depth_areas[0])
        lake_energy_deltas = lake_energy_deltas[1:]
        #calculate sum of energy flux into or out of the lake at each timestep
        # print("dates ", dates[0,1:6])
        lake_energy_fluxes = calculate_energy_fluxes(phys[start_index,:,:], outputs[start_index,:], combine_days)
        ### can use this to plot energy delta and flux over time to see if they line up
        doy = np.array([datetime.datetime.combine(date.fromordinal(x), datetime.time.min).timetuple().tm_yday  for x in dates[start_index,:]])
        doy = doy[1:-1]
        diff_vec = (lake_energy_deltas - lake_energy_fluxes).abs_()
        

        #actual ice
        # diff_vec = diff_vec[np.where((phys[0,1:-1,9] == 0))[0]]
        diff_vec = diff_vec[np.where((phys[0,1:-1,-1] == 0))[0]] #change from 9 to -1 not verified
        # #compute difference to be used as penalty
        if diff_vec.size() == torch.Size([0]):
            diff_per_set[i] = 0
        else:
            diff_per_set[i] = diff_vec.mean()
    if use_gpu:
        diff_per_set = diff_per_set.cuda()
    diff_per_set = torch.clamp(diff_per_set - ec_threshold, min=0)
    print(diff_per_set.mean())
    return diff_per_set.mean()

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


def calculate_lake_energy(temps, densities, depth_areas):
    #calculate the total energy of the lake for every timestep
    #sum over all layers the (depth cross-sectional area)*temp*density*layer_height)
    #then multiply by the specific heat of water 
    dz = 0.5 #thickness for each layer, hardcoded for now
    cw = 4186 #specific heat of water
    energy = torch.empty_like(temps[0,:])
    n_depths = depth_areas.size()[0]
    depth_areas = depth_areas.view(n_depths,1).expand(n_depths, temps.size()[1])
    energy = torch.sum(depth_areas*temps*densities*0.5*cw,0)
    return energy


def calculate_lake_energy_deltas(energies, combine_days, surface_area):
    #given a time series of energies, compute and return the differences
    # between each time step, or time step interval (parameter @combine_days)
    # as specified by parameter @combine_days
    energy_deltas = torch.empty_like(energies[0:-combine_days])
    time = 86400 #seconds per day
    # surface_area = 39865825
    energy_deltas = (energies[1:] - energies[:-1])/(time*surface_area)
    # for t in range(1, energy_deltas.size()[0]):
    #     energy_deltas[t-1] = (energies[t+combine_days] - energies[t])/(time*surface_area) #energy difference converted to W/m^2
    return energy_deltas





def calculate_energy_fluxes(phys, surf_temps, combine_days):
    # print("surface_depth = ", phys[0:5,1])
    fluxes = torch.empty_like(phys[:-combine_days-1,0])

    time = 86400 #seconds per day
    surface_area = 39865825 

    e_s = 0.985 #emissivity of water, given by Jordan
    alpha_sw = 0.07 #shortwave albedo, given by Jordan Read
    alpha_lw = 0.03 #longwave, albeda, given by Jordan Read
    sigma = 5.67e-8 #Stefan-Baltzmann constant
    R_sw_arr = phys[:-1,2] + (phys[1:,2]-phys[:-1,2])/2
    R_lw_arr = phys[:-1,3] + (phys[1:,3]-phys[:-1,3])/2
    R_lw_out_arr = e_s*sigma*(torch.pow(surf_temps[:]+273.15, 4))
    R_lw_out_arr = R_lw_out_arr[:-1] + (R_lw_out_arr[1:]-R_lw_out_arr[:-1])/2

    air_temp = phys[:-1,4] 
    air_temp2 = phys[1:,4]
    rel_hum = phys[:-1,5]
    rel_hum2 = phys[1:,5]
    ws = phys[:-1, 6]
    ws2 = phys[1:,6]
    t_s = surf_temps[:-1]
    t_s2 = surf_temps[1:]
    E = phys_operations.calculate_heat_flux_latent(t_s, air_temp, rel_hum, ws)
    H = phys_operations.calculate_heat_flux_sensible(t_s, air_temp, rel_hum, ws)
    E2 = phys_operations.calculate_heat_flux_latent(t_s2, air_temp2, rel_hum2, ws2)
    H2 = phys_operations.calculate_heat_flux_sensible(t_s2, air_temp2, rel_hum2, ws2)
    E = (E + E2)/2
    H = (H + H2)/2

    #test
    fluxes = (R_sw_arr[:-1]*(1-alpha_sw) + R_lw_arr[:-1]*(1-alpha_lw) - R_lw_out_arr[:-1] + E[:-1] + H[:-1])


    return fluxes

def calculate_energy_fluxes_manylakes(phys, surf_temps, combine_days):
    fluxes = torch.empty_like(phys[:-combine_days-1,0])

    time = 86400 #seconds per day
    surface_area = 39865825 

    e_s = 0.985 #emissivity of water, given by Jordan
    alpha_sw = 0.07 #shortwave albedo, given by Jordan Read
    alpha_lw = 0.03 #longwave, albeda, given by Jordan Read
    sigma = 5.67e-8 #Stefan-Baltzmann constant
    R_sw_arr = phys[:-1,1] + (phys[1:,1]-phys[:-1,1])/2
    R_lw_arr = phys[:-1,2] + (phys[1:,2]-phys[:-1,2])/2
    R_lw_out_arr = e_s*sigma*(torch.pow(surf_temps[:]+273.15, 4))
    R_lw_out_arr = R_lw_out_arr[:-1] + (R_lw_out_arr[1:]-R_lw_out_arr[:-1])/2

    air_temp = phys[:-1,3] 
    air_temp2 = phys[1:,3]
    rel_hum = phys[:-1,4]
    rel_hum2 = phys[1:,4]
    ws = phys[:-1, 5]
    ws2 = phys[1:,5]
    t_s = surf_temps[:-1]
    t_s2 = surf_temps[1:]
    E = phys_operations.calculate_heat_flux_latent(t_s, air_temp, rel_hum, ws)
    H = phys_operations.calculate_heat_flux_sensible(t_s, air_temp, rel_hum, ws)
    E2 = phys_operations.calculate_heat_flux_latent(t_s2, air_temp2, rel_hum2, ws2)
    H2 = phys_operations.calculate_heat_flux_sensible(t_s2, air_temp2, rel_hum2, ws2)
    E = (E + E2)/2
    H = (H + H2)/2

    #test
    fluxes = (R_sw_arr[:-1]*(1-alpha_sw) + R_lw_arr[:-1]*(1-alpha_lw) - R_lw_out_arr[:-1] + E[:-1] + H[:-1])


    return fluxes

def getHypsographyManyLakes(path, lakename, depths):
    my_path = os.path.abspath(os.path.dirname(__file__))
    if not os.path.exists(os.path.join(my_path, path)):
        print("no hypsography file")
        return None
    depth_areas = pd.read_csv(os.path.join(my_path, path), header=0, index_col=0, squeeze=True).to_dict()

    if len(depth_areas) < 3:
        #new try 
        avail_depths = np.array(list(depth_areas.keys()))
        avail_areas = np.array(list(depth_areas.values())) 
        sort_ind = np.argsort(avail_depths)
        avail_depths = avail_depths[sort_ind]
        avail_areas = avail_areas[sort_ind]

        f = interpolate.interp1d(avail_depths, avail_areas,  fill_value="extrapolate")
        new_depth_areas = np.array([f(x) for x in depths])
        return new_depth_areas

    else:
        #old try
        tmp = {}
        total_area = 0
        for key, val in depth_areas.items():
            total_area += val

        for depth in depths:
            #find depth with area that is closest
            depth_w_area = min(list(depth_areas.keys()), key=lambda x:abs(x-depth))
            tmp[depth] = depth_areas[depth_w_area]
        depth_areas = {}

        for k, v in tmp.items():
            total_area += v

        for k, v in tmp.items():
            depth_areas[k] = tmp[k] 


        return np.sort(-np.array([list(depth_areas.values())]))*-1


def getHypsography(lakename, depths, debug=False, path=None):
    my_path = os.path.abspath(os.path.dirname(__file__))
    depth_areas = []
    if path is not None:
        depth_areas = pd.read_csv(path, header=0, index_col=0, squeeze=True).to_dict()
    else:
        depth_areas = pd.read_csv(os.path.join(my_path, '../../data/raw/'+lakename+'/'+lakename+'_hypsography.csv'), header=0, index_col=0, squeeze=True).to_dict()
    tmp = {}
    total_area = 0
    for key, val in depth_areas.items():
        total_area += val

    for depth in depths:
        #find depth with area that is closest
        depth_w_area = min(list(depth_areas.keys()), key=lambda x:abs(x-depth))
        tmp[depth] = depth_areas[depth_w_area]
    depth_areas = {}

    for k, v in tmp.items():
        total_area += v

    for k, v in tmp.items():
        depth_areas[k] = tmp[k] 

    return np.sort(-np.array([list(depth_areas.values())]))*-1

#define LSTM model class
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, use_gpu):
        super(LSTM, self).__init__()
        self.use_gpu = use_gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size = input_size, hidden_size=hidden_size, batch_first=True) #batch_first=True?
        self.out = nn.Linear(hidden_size, 1) #1?
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=0):
            # initialize both hidden layers
            if batch_size == 0:
                batch_size = self.batch_size
            ret = (xavier_normal_(torch.empty(1, batch_size, self.hidden_size)),
                    xavier_normal_(torch.empty(1, batch_size, self.hidden_size)))
            # print("hidden layer initialized: ", ret)
            # if data_parallel: #TODO??
            #     ret = (xavier_normal_(torch.empty(1, math.ceil(self.batch_size/2), self.hidden_size/2)),
            #         xavier_normal_(torch.empty(1, math.floorself.batch_size, math.floor(self.hidden_size/2))))
            if self.use_gpu:
                item0 = ret[0].cuda(non_blocking=True)
                item1 = ret[1].cuda(non_blocking=True)
                ret = (item0,item1)
            return ret
    def init_hidden_test(self, batch_size=0):
        # initialize both hidden layers
        if batch_size == 0:
            batch_size = self.batch_size
        torch.manual_seed(0)
        if self.use_gpu:
            torch.cuda.manual_seed_all(0)
        # print("epoch ", epoch+1)
        ret = (xavier_normal_(torch.empty(1, batch_size, self.hidden_size)),
                xavier_normal_(torch.empty(1, batch_size, self.hidden_size)))
        # print("hidden layer initialized: ", ret)
        # if data_parallel: #TODO??
        #     ret = (xavier_normal_(torch.empty(1, math.ceil(self.batch_size/2), self.hidden_size/2)),
        #         xavier_normal_(torch.empty(1, math.floorself.batch_size, math.floor(self.hidden_size/2))))
        if self.use_gpu:
            item0 = ret[0].cuda(non_blocking=True)
            item1 = ret[1].cuda(non_blocking=True)
            ret = (item0,item1)
        return ret
    
    def forward(self, x, hidden):
        # print("X size is {}".format(x.size()))
        self.lstm.flatten_parameters()

        x = x.float()
        x, hidden = self.lstm(x, self.hidden)
        self.hidden = hidden
        x = self.out(x)
        return x, hidden

class ContiguousBatchSampler(object):
    def __init__(self, batch_size, n_batches):
        # print("batch size", batch_size)
        # print("n batch ", n_batches)
        self.sampler = torch.randperm(n_batches)
        self.batch_size = batch_size

    def __iter__(self):
        for idx in self.sampler:
            yield torch.arange(idx*self.batch_size, (idx+1)*self.batch_size, dtype=torch.long)

    def __len__(self):
        return len(self.sampler) // self.batch_size

class RandomContiguousBatchSampler(object):
    def __init__(self, n_dates, seq_length, batch_size, n_batches):
        # note: batch size = n_depths here
        #       n_dates = number of all* sequences (*yhat driver dataset)
        # high = math.floor((n_dates-seq_length)/batch_size) dated and probably wrong
        high = math.floor(n_dates/batch_size)

        self.sampler = torch.randint_like(torch.empty(n_batches), low=0, high=high)        
        self.batch_size = batch_size

    def __iter__(self):
        for idx in self.sampler:
            # yield torch.arange(idx*self.batch_size, (idx+1)*self.batch_size, dtype=torch.long) #old
            yield torch.arange(idx*self.batch_size, (idx+1)*self.batch_size, dtype=torch.long) 

    def __len__(self):
        return len(self.sampler) // self.batch_size

def parseMatricesFromSeqs(pred, targ, dates, n_tst_dates, u_dates):
    #format an array of sequences into one [depths x timestep] matrix
    assert pred.shape[0] == targ.shape[0]
    n_seq = pred.shape[0]
    seq_len = int(pred.shape[1])
    out_mat = np.empty((n_tst_dates))
    out_mat[:] = np.nan
    lab_mat = np.empty((n_tst_dates))
    lab_mat[:] = np.nan
    for i in np.arange(n_seq-1,-1,-1):
        #for each sequence
        if i >= dates.shape[0]:
            print("more sequences than dates")
            continue
        #find depth index
        
        #find date index
        if np.isnat(dates[i,0]):
            print("not a time found")
            continue
        if len(np.where(u_dates == dates[i,0])[0]) == 0:
            print("invalid date")
            continue
        date_ind = np.where(u_dates == dates[i,0])[0][0]
        if out_mat[date_ind:].shape[0] < seq_len:
            sizeToCopy = out_mat[date_ind:].shape[0] #this is to not copy data beyond test dates
            out_mat[date_ind:] = pred[i,:sizeToCopy]
            lab_mat[date_ind:] = targ[i,:sizeToCopy]
        else:
            indices = np.isfinite(targ[i,:])
            out_mat[date_ind:date_ind+seq_len] = pred[i,:]
            lab_mat[date_ind:date_ind+seq_len][indices] = targ[i,:][indices]
                
    return (out_mat, lab_mat)


def transformTempToDensity(temp, use_gpu):
    # print(temp)
    #converts temperature to density
    #parameter:
        #@temp: single value or array of temperatures to be transformed
    densities = torch.empty_like(temp)
    if use_gpu:
        temp = temp.cuda()
        densities = densities.cuda()
    # return densities
    # print(densities.size()
    # print(temp.size())
    densities[:] = 1000*(1-((temp[:]+288.9414)*torch.pow(temp[:] - 3.9863,2))/(508929.2*(temp[:]+68.12963)))
    # densities[:] = 1000*(1-((temp[:]+288.9414)*torch.pow(temp[:] - 3.9863))/(508929.2*(temp[:]+68.12963)))
    # print("DENSITIES")
    # for i in range(10):
    #     print(densities[i,i])

    return densities

#Iterator through multiple dataloaders
class MyIter(object):
  """An iterator."""
  def __init__(self, my_loader):
    self.my_loader = my_loader
    self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]
    # print("init", self.loader_iters)

  def __iter__(self):
    return self

  def __next__(self):
    # When the shortest loader (the one with minimum number of batches)
    # terminates, this iterator will terminates.
    # The `StopIteration` raised inside that shortest loader's `__next__`
    # method will in turn gets out of this `__next__` method.
    # print("next",     print(self.loader_iters))
    batches = [loader_iter.next() for loader_iter in self.loader_iters]
    return self.my_loader.combine_batch(batches)

  # Python 2 compatibility
  next = __next__

  def __len__(self):
    return len(self.my_loader)

#wrapper class for multiple dataloaders
class MultiLoader(object):
  """This class wraps several pytorch DataLoader objects, allowing each time 
  taking a batch from each of them and then combining these several batches 
  into one. This class mimics the `for batch in loader:` interface of 
  pytorch `DataLoader`.
  Args: 
    loaders: a list or tuple of pytorch DataLoader objects
  """
  def __init__(self, loaders):
    self.loaders = loaders

  def __iter__(self):
    return MyIter(self)

  def __len__(self):
    l =  min([len(loader) for loader in self.loaders])
    # print(l)
    return l

  # Customize the behavior of combining batches here.
  def combine_batch(self, batches):
    return batches


def xavier_normal_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std})` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return _no_grad_normal_(tensor, 0., std)

