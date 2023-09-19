import torch
from torch.functional import F
import torch.nn as nn
from torch.nn.parameter import Parameter

import os
import numpy as np
from math import sqrt
from scipy import stats

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SpatialGroupEnhance_for_1D(nn.Module):
    def __init__(self, groups = 32):
        super(SpatialGroupEnhance_for_1D, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1))
        self.sig      = nn.Sigmoid()
    
    def forward(self, x): # (b, c, h)
        b, c, h = x.size()
        x = x.view(b * self.groups, -1, h)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h)
        x = x * self.sig(t)
        x = x.view(b, c, h)
        return x

class DAT_cnn(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, hidden_dim, dropout_rate,
                 alpha, n_heads, bilstm_layers=2, protein_vocab=26,
                 smile_vocab=63):
        super(DAT_cnn, self).__init__()

        seq_len = 100
        tar_len = 2000

        self.is_bidirectional = True
        # drugs
        
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.bilstm_layers = bilstm_layers
        self.n_heads = n_heads

        # SMILES
        self.smiles_vocab = smile_vocab
        self.smiles_embed = nn.Embedding(smile_vocab + 3, embedding_dim, padding_idx=smile_vocab)

        self.is_bidirectional = True
        self.smiles_input_fc = nn.Linear(embedding_dim, lstm_dim)
        self.smiles_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                   bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln1 = torch.nn.LayerNorm(lstm_dim * 2)
        
        self.enhance1= SpatialGroupEnhance_for_1D(groups=20)

        # self.out_attentions3 = LinkAttention(hidden_dim, n_heads)

        # protein
        self.protein_vocab = protein_vocab
        self.protein_embed = nn.Embedding(protein_vocab + 3, embedding_dim, padding_idx=protein_vocab)
        self.is_bidirectional = True
        self.protein_input_fc = nn.Linear(embedding_dim, lstm_dim)

        self.protein_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                    bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln2 = torch.nn.LayerNorm(lstm_dim * 2)

        self.enhance2 = SpatialGroupEnhance_for_1D(groups=200)

        self.protein_head_fc = nn.Linear(lstm_dim * n_heads, lstm_dim)
        self.protein_out_fc = nn.Linear(2 * lstm_dim, hidden_dim)
        # self.out_attentions2 = LinkAttention(hidden_dim, n_heads)

        self.concat_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                   bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.concat_ln = torch.nn.LayerNorm(lstm_dim * 2)

        self.smiles_cnn = nn.Conv1d(seq_len, int(seq_len / 2), kernel_size=3, padding=1)
        self.protein_cnn = nn.Conv1d(tar_len, int(tar_len / 2), kernel_size=3, padding=1)
        self.combine_cnn = nn.Conv1d(seq_len + tar_len, int((seq_len + tar_len) / 2), kernel_size=3, padding=1)

        self.smiles_cnn_fc = nn.Linear(seq_len * lstm_dim, lstm_dim * 2)

        self.protein_cnn_fc = nn.Linear(tar_len * lstm_dim, lstm_dim * 2)
        self.combine_cnn_fc = nn.Linear((seq_len + tar_len) * lstm_dim, lstm_dim * 2)

        self.total_cnn_fc = nn.Linear((seq_len + tar_len) * lstm_dim * 2, lstm_dim * 2)

        # link
        # self.out_attentions = LinkAttention(hidden_dim, n_heads)
        self.out_fc1 = nn.Linear(hidden_dim * 4, hidden_dim * n_heads)
        self.out_fc2 = nn.Linear(hidden_dim * n_heads, hidden_dim * 2)
        self.out_fc3 = nn.Linear(hidden_dim * 2, 1)
        self.layer_norm = nn.LayerNorm(lstm_dim * 2)

    def forward(self, data, reset=False):
        protein, smiles = data[1].to(device), data[0].to(device)
        smiles_lengths = data[-2].to(device)
        protein_lengths = data[-1].to(device)
        batchsize = len(protein)

        smiles = self.smiles_embed(smiles)
        smiles = self.smiles_input_fc(smiles)
        smiles = self.enhance1(smiles)  

        protein = self.protein_embed(protein) 
        protein = self.protein_input_fc(protein) 
        protein = self.enhance2(protein) 

        concat_emb = torch.cat((smiles, protein), dim=1)
        concat_emb, _ = self.concat_lstm(concat_emb)
        concat_emb = self.concat_ln(concat_emb)

        smiles, _ = self.smiles_lstm(smiles)
        smiles = self.ln1(smiles)

        protein, _ = self.protein_lstm(protein)
        protein = self.ln2(protein)

        smiles = self.relu(self.smiles_cnn(smiles))

        protein = self.relu(self.protein_cnn(protein))

        concat_emb = self.relu(self.combine_cnn(concat_emb))

        total_emb = torch.cat((smiles, protein, concat_emb), dim=1)

        smiles = self.relu(self.smiles_cnn_fc(smiles.view(smiles.shape[0], -1)))
        protein = self.relu(self.protein_cnn_fc(protein.view(protein.shape[0], -1)))
        concat_emb = self.relu(self.combine_cnn_fc(concat_emb.view(concat_emb.shape[0], -1)))
        total_emb = self.total_cnn_fc(total_emb.view(total_emb.shape[0], -1))

        out_cat = torch.cat((smiles, protein, concat_emb, total_emb), dim=-1)
        out = self.dropout(self.relu(self.out_fc1(out_cat))) 
        out = self.dropout(self.relu(self.out_fc2(out))) 
        out = self.out_fc3(out).squeeze()

        return out