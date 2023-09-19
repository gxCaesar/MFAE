
dataset_name = "CRC_5_fold"
model_name = "MFAE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np

from model import DAT_cnn

from dataset import *
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ci_list = []
mse_list = []
rm2_list = []
pearson_list = []
spearman_list = []


for fold in range(1,6):
    model1 = DAT_cnn(128, 64, 128, 0.1, 0.1, 8).to(device)
    model_file = './Model/'+dataset_name+"_"+model_name+"_fold_"+str(fold) + "_esamble_" +str(1)+'.pt'
    model1.load_state_dict(torch.load(model_file))
    model1.eval()
    model2 = DAT_cnn(128, 64, 128, 0.1, 0.1, 8).to(device)
    model_file = './Model/'+dataset_name+"_"+model_name+"_fold_"+str(fold) + "_esamble_" +str(2)+'.pt'
    model2.load_state_dict(torch.load(model_file))
    model2.eval()
    model3 = DAT_cnn(128, 64, 128, 0.1, 0.1, 8).to(device)
    model_file = './Model/'+dataset_name+"_"+model_name+"_fold_"+str(fold) + "_esamble_" +str(3)+'.pt'
    model3.load_state_dict(torch.load(model_file))
    model3.eval()
    
    test_dataset_path = "./Data/fold_{}_test.csv".format(fold)
    test_dataset = DTADataset(test_dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    G1, P1 = pre_predicting(model1, test_loader)
    G2, P2 = pre_predicting(model2, test_loader)
    G3, P3 = pre_predicting(model3, test_loader)
    
    G = G1
    P = (P1+P2+P3)/3
    cindex,rm2,mse,pearson,spearman= calculate_metrics_and_return(G, P, test_loader)
    ci_list.append(cindex)
    rm2_list.append(rm2)
    mse_list.append(mse)
    pearson_list.append(pearson)
    spearman_list.append(spearman)
df = pd.DataFrame({'cindex':ci_list,'rm2':rm2_list,'mse':mse_list,'pearson':pearson_list,'spearman':spearman_list})
df.to_csv("./Result/"+dataset_name+"_"+model_name+"_esamble_resule.csv",index=False)