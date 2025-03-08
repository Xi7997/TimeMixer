# -*- coding: UTF-8 -*-
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import csv

class Weather_Dataset(Dataset):
    def __init__(self, data_path, seq_len=96, pred_len=96):
        self.data = pd.read_csv(data_path)

        self.data = self.data.iloc[:, 1:]

        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(self.data)

        self.seq_len = seq_len
        self.pred_len = pred_len

        self.X, self.Y = self.sliding_windows(self.data, seq_len, pred_len)

    def sliding_windows(self, data, seq_len, pred_len):
        x = []
        y = []
        for i in range(len(data) - seq_len - pred_len):
            _x = data[i:(i + seq_len)]
            _y = data[(i + seq_len):(i + seq_len + pred_len)]
            x.append(_x)
            y.append(_y)
        return x, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)


def get_dataloader(data_path, seq_len=96, pred_len=96, batch_size=32):
    dataset = Weather_Dataset(data_path, seq_len, pred_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


