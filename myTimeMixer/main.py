# -*- coding: UTF-8 -*-
import numpy as np
import torch
from Dataloader import Weather_Dataset, get_dataloader
import os
import csv

data_path = '../dataset/weather/weather.csv'
seq_len = 96
pred_len = 96

train_loader = get_dataloader(data_path, seq_len, pred_len, batch_size=32)

# 提取X，Y
for i, (X, Y) in enumerate(train_loader):
    print(X.shape,Y.shape)
    break





