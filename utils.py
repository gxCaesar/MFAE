import yaml
import argparse
import numpy as np
import torch
from os.path import dirname, join, exists
import pandas as pd
from math import sqrt
from sklearn.metrics import average_precision_score
from scipy import stats
import random
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, train_loader ,criterion , optimizer):
    model.train()
    for i,data in enumerate(tqdm(train_loader)):
        y_predict = model(data)
        loss = criterion(y_predict, data[-3].float().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def pre_predicting(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in tqdm(loader):
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data[-3].view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def get_aupr(Y, P, threshold=7.0):
    Y = np.where(Y >= 7.0, 1, 0)
    P = np.where(P >= 7.0, 1, 0)
    aupr = average_precision_score(Y, P)
    return aupr


def get_cindex(Y, P):
    summ = 0
    pair = 0
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])
    if pair is not 0:
        return summ / pair
    else:
        return 0


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]
    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult
    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def get_rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def get_pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def get_spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def calculate_metrics_and_return(Y, P, dataset='kiba'):
    cindex = get_ci(Y, P)
    rm2 = get_rm2(Y, P) 
    mse = get_mse(Y, P)
    pearson = get_pearson(Y, P)
    spearman = get_spearman(Y, P)
    rmse = get_rmse(Y, P)

    print('metrics for ', dataset)
    print('cindex:', cindex)
    print('rm2:', rm2)
    print('mse:', mse)
    print('pearson', pearson)
    print('spearman',spearman)
    return cindex,rm2,mse,pearson,spearman

def get_ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse





def getdata_from_csv(fname, maxlen=512):
    df = pd.read_csv(fname)
    smiles = list(df['compound_iso_smiles'])
    protein = list(df['target_sequence'])
    affinity = list(df['affinity'])
    return smiles, protein, affinity



    

def collate(args):
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    return x0, x1, torch.stack(y, 0)



class Alphabets():
    def __init__(self, chars, encoding=None, missing=63):
        self.chars = np.frombuffer(chars, dtype='uint8')
        self.size = len(self.chars)
        self.encoding = np.zeros(256, dtype='uint8') + missing
        if encoding == None:
            self.encoding[self.chars] = np.arange(self.size)
        else:
            self.encoding[self.chars] = encoding
            
    def encode(self, s):
        s = np.frombuffer(s, dtype='uint8')
        return self.encoding[s]
    
class AminoAcid(Alphabets):
    def __init__(self):
        chars = b'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        super(AminoAcid, self).__init__(chars)
        
class Smiles(Alphabets):
    def __init__(self):
        chars = b'#%)(+-.1032547698=ACBEDGFIHKMLONPSRUTWVY[Z]_acbedgfihmlonsruty@'
        super(Smiles, self).__init__(chars)



class LoadFromFile(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config.keys():
                if key not in namespace:
                    raise ValueError(f"Unknown argument in config file: {key}")
            namespace.__dict__.update(config)
        else:
            raise ValueError("Configuration file must end with yaml or yml")


class LoadFromCheckpoint(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        hparams_path = join(dirname(values), "hparams.yaml")
        if not exists(hparams_path):
            print(
                "Failed to locate the checkpoint's hparams.yaml file. Relying on command line args."
            )
            return
        with open(hparams_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for key in config.keys():
            if key not in namespace and key != "prior_args":
                raise ValueError(f"Unknown argument in the model checkpoint: {key}")
        namespace.__dict__.update(config)
        namespace.__dict__.update(load_model=values)


def save_argparse(args, filename, exclude=None):
    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [exclude]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]
        yaml.dump(args, open(filename, "w"))
    else:
        raise ValueError("Configuration file should end with yaml or yml")


def number(text):
    if text is None or text == "None":
        return None

    try:
        num_int = int(text)
    except ValueError:
        num_int = None
    num_float = float(text)

    if num_int == num_float:
        return num_int
    return num_float