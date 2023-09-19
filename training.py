import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import re
import csv
import random
import copy
from utils import *
import os
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from model import DAT_cnn
from dataset import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


LR = 1e-3
NUM_EPOCHS = 200
dataset_name = "CRC_5_fold"
model_name = "MFAE"


ci_list = []
mse_list = []
rm2_list = []
pearson_list = []
spearman_list = []


num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True,random_state=0)
batch_size = 128





for esamble_idx in range(1,4):
    for fold in range(1,6):
        model = DAT_cnn(128, 64, 128, 0.1, 0.1, 8).to(device)

        model_file_name = './Model/'+dataset_name+"_"+model_name+"_fold_"+str(fold) + "_esamble_" +str(esamble_idx)+'.pt'

        print(f"Fold {fold}")

        train_dataset_path = "./Data/bagging_data/fold_{}_train_sample_{}.csv".format(fold,esamble_idx)
        val_dataset_path = "./Data/fold_{}_val.csv".format(fold)
        test_dataset_path = "./Data/fold_{}_test.csv".format(fold)
        train_dataset = DTADataset(train_dataset_path)
        val_dataset = DTADataset(val_dataset_path)
        test_dataset = DTADataset(test_dataset_path)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print('Train size:', len(train_dataset))
        print('Valid size:', len(val_dataset))
        print('Test size:', len(test_dataset))

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50, eta_min=5e-4, last_epoch=-1)

        best_mse = 1000
        best_test_mse = 1000
        best_epoch = -1
        best_test_epoch = -1
        start = time.time()

        for epoch in range(NUM_EPOCHS):
            train(model , train_loader,loss_fn,optimizer)
            G, P = pre_predicting(model, val_loader)
            val1 = get_mse(G, P)
            if val1 < best_mse:
                best_mse = val1
                best_epoch = epoch + 1
                if model_file_name is not None:
                    torch.save(model.state_dict(), model_file_name)
                print('mse improved at epoch ', best_epoch, '; best_mse', best_mse)
            schedule.step()

        ## start testing
        print(model_file_name)
        save_model = torch.load(model_file_name)
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        G, P = pre_predicting(model, test_loader)
        cindex,rm2,mse,pearson,spearman= calculate_metrics_and_return(G, P, test_loader)
        ci_list.append(cindex)
        rm2_list.append(rm2)
        mse_list.append(mse)
        pearson_list.append(pearson)
        spearman_list.append(spearman)
    df = pd.DataFrame({'cindex':ci_list,'rm2':rm2_list,'mse':mse_list,'pearson':pearson_list,'spearman':spearman_list})
    df.to_csv('./Result/'+dataset_name+'_'+model_name+ '_'+str(esamble_idx)+'.csv',index=False)
    ci_list = []
    mse_list = []
    rm2_list = []
    pearson_list = []
    spearman_list = []