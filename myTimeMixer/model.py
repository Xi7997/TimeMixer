# -*- coding: UTF-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from Autoformer_EncDec import series_decomp
from Embed import DataEmbedding_wo_pos
from SeasonTrendMixing import MultiScaleSeasonMixing, MultiScaleTrendMixing
from FMM import FutureMultipredictorMixing
DEBUG = False
class TimeMixer(nn.Module):
    def __init__(self, input_dim=21, seq_len=96, pred_len=96, d_model=16, down_sampling_layers=3, down_sampling_window=2):
        super(TimeMixer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.embedding = DataEmbedding_wo_pos(input_dim, d_model, embed_type='fixed', freq='h', dropout=0.1)

        # 输入映射（输入维度，嵌入维度）
        self.input_layer = nn.Linear(input_dim, d_model)

        # 降采样downsample参数(3层降采样，每次除以2)
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.decomposition = series_decomp(25)

        self.mixing_multi_scale_season = MultiScaleSeasonMixing(seq_len)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(seq_len)
        self.Predictor = FutureMultipredictorMixing(seq_len, pred_len, d_model, input_dim, down_sampling_layers+1)

        # 多层降采样
        self.down_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len // (down_sampling_window ** i), seq_len // (down_sampling_window ** (i + 1))),
                nn.GELU(),
                nn.Linear(seq_len // (down_sampling_window ** (i + 1)), seq_len // (down_sampling_window ** (i + 1))),
            )
            for i in range(down_sampling_layers)
        ])

        # 残差链接层
        # self.residual_layers = nn.ModuleList([
        #     nn.Linear(seq_len // (down_sampling_window ** (i)), seq_len // (down_sampling_window ** (i)))
        #     for i in range(down_sampling_layers)
        # ])


        # self.upsample_layer = nn.Upsample(size=96, mode='linear', align_corners=True)
        # self.expend_t = nn.Linear(12, 96)
        # self.output_layer = nn.Linear(16, 21)

        # # 输出预测层
        # # self.output_layer = nn.Linear(d_model, pred_len)
        # self.fc1 = nn.Linear(d_model, 64)  # 升维
        # self.fc2 = nn.Linear(64, 32)  # 降维
        # self.fc3 = nn.Linear(32, input_dim)

    def PastDecomposableMixing(self, X_list):
        season_list = []
        trend_list = []

        for X in X_list:
            # 分解
            season,trend = self.decomposition(X)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        new_season_list = self.mixing_multi_scale_season(season_list)
        new_trend_list = self.mixing_multi_scale_trend(trend_list)

        X1_list = []

        for X, new_season, new_trend in zip(X_list, new_season_list, new_trend_list):
            length = X.shape[1]
            new = new_season + new_trend
            X1_list.append(new[:, :length, :])
        return X1_list

    def forward(self, x):
        B, T, C = x.shape  # Batch, 时间步, 通道数
        # step1: 多尺度时间序列输入
        X = self.embedding(x, None)
        X_list = [X]
        for layer in self.down_sampling_layers:
            X = layer(X.permute(0, 2, 1)).permute(0, 2, 1)
            X_list.append(X)
        if DEBUG:
            for i,s in enumerate(X_list):
                print(f"after down_sampling_layers {i}: ", s.shape)

        # step2: 多尺度混合
        X1_list = self.PastDecomposableMixing(X_list)
        if DEBUG:
            for i,s in enumerate(X1_list):
                print(f"after PastDecomposableMixing {i}: ", s.shape)
        #step3: 混合预测
        # print("11111")
        y_pre = self.Predictor(X1_list)
        # print("22222")
        if DEBUG:
            print("Y-pre: ", y_pre)
            print("after Predictor: ", y_pre.shape)

        return y_pre

    # def forward(self, x):
    #     B, T, C = x.shape  # Batch, 时间步, 通道数
    #     # if DEBUG:
    #     #     print("before embedding", x.shape)
    #     # # print("after permute", x.shape)
    #     #
    #     # # 1. embedding嵌入 21->16
    #     # x = self.embedding(x, None)  # (B, T, d)
    #     # if DEBUG:
    #     #     print("after embedding", x.shape)
    #     #
    #     # # 2. 多尺度融合（multi-scale）
    #     # x_list = [x]
    #     # for layer in self.down_sampling_layers:
    #     #     x = layer(x.permute(0, 2, 1)).permute(0, 2, 1)
    #     #     x_list.append(x)
    #     # if DEBUG:
    #     #     for i,s in enumerate(x_list):
    #     #         print(f"after down_sampling_layers {i}: ", s.shape)
    #     #
    #     # out_high = x_list[0]
    #     # out_low = x_list[1]
    #     #
    #     # # 残差连接
    #     # for i in range(len(x_list) - 1):
    #     #     if DEBUG:
    #     #         print("out_high", out_high.shape)
    #     #         print("out_low", out_low.shape)
    #     #     out_low_res = self.down_sampling_layers[i](out_high.permute(0, 2, 1)).permute(0, 2, 1)
    #     #     out_low = out_low + out_low_res
    #     #     out_high = out_low
    #     #     if i + 2 <= len(x_list) - 1:
    #     #         out_low = x_list[i + 2]
    #     # x = out_high
    #     #
    #     # # 3. predict
    #     # x = self.expend_t(x.permute(0, 2, 1)).permute(0, 2, 1)
    #     # if DEBUG:
    #     #     print("after upsample", x.shape)
    #     # x = self.output_layer(x)
    #     # if DEBUG:
    #     #     print("after output_layer", x.shape)






