pro_pad_vocab = 26
pro_cls_vocab = 27
pro_end_vocab = 28

smile_vocab = 63
smi_cls_vocab = 64
smi_end_vocab = 65

import torch
import os
import numpy as np
import pandas as pd
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
from tqdm import tqdm

from utils import Smiles,AminoAcid

Alphabet = AminoAcid()
smilebet = Smiles()

class DTADataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, seq_len=100, tar_max_len=2000):
        df = pd.read_csv(dataset_path)
        smiles = set(df['compound_iso_smiles'])
        protein = set(df['target_sequence'])
        affinity = list(df['affinity'])

        sm_emb = dict()
        sm_len = dict()
        for sm in smiles:
            emb = sm.encode('utf-8').upper()
            emb = smilebet.encode(emb)
            emb = list(emb)

            if len(emb) > seq_len - 2:
                emb = emb[:seq_len - 2]
            emb = [smi_cls_vocab] + emb + [smi_end_vocab]
            sm_len[sm] = len(emb)
            if seq_len > len(emb):
                padding = [smile_vocab] * (seq_len - len(emb))
                emb = emb + padding

            emb = torch.tensor(emb).long()
            sm_emb[sm] = emb

        tar_emb = dict()
        tar_len = dict()
        for tar in protein:
            emb = tar.encode('utf-8').upper()
            emb = Alphabet.encode(emb)
            emb = list(emb)

            if len(emb) > tar_max_len - 2:
                emb = emb[:tar_max_len - 2]
            emb = [pro_cls_vocab] + emb + [pro_end_vocab]

            tar_len[tar] = len(emb)

            if tar_max_len > len(emb):
                padding = [pro_pad_vocab] * (tar_max_len - len(emb))
                emb = emb + padding

            emb = torch.tensor(emb).long()
            tar_emb[tar] = emb

        self.tar_len = tar_len
        self.seq_len = seq_len

        self.smiles = []
        self.targets = []
        self.label = []
        self.sm_len = []
        self.tar_len = []

        for i in tqdm(range(len(df))):
            sm = df.loc[i, 'compound_iso_smiles']
            seq = df.loc[i, 'target_sequence']
            label = df.loc[i, 'affinity']

            smiles_emb = sm_emb[sm]
            target_emb = tar_emb[seq]

            smiles_len = sm_len[sm]
            target_len = tar_len[seq]

            self.smiles.append(smiles_emb)
            self.targets.append(target_emb)
            self.label.append(label)
            self.sm_len.append(smiles_len)
            self.tar_len.append(target_len)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        return [self.smiles[item], self.targets[item], self.label[item], self.sm_len[item], self.tar_len[item]]


class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y,smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])