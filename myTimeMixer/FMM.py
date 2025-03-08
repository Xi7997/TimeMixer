# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
class FutureMultipredictorMixing(nn.Module):
    def __init__(self, seq_len=96, pred_len=96, d_model=16, c_out=21, num_scales=4):
        super(FutureMultipredictorMixing, self).__init__()

        # 不同尺度的预测层
        self.predict_layers = nn.ModuleList([
            nn.Linear(seq_len // (2 ** i), pred_len) for i in range(num_scales)
        ])

        # 输出投影层，把 d_model -> c_out
        self.projection_layer = nn.Linear(d_model, c_out)

    def forward(self, x_list):
        """
        x_list: 包含多个尺度的输入
        """
        pred_list = []

        for i, x in enumerate(x_list):
            # 线性预测未来 pred_len 个时间步
            pred = self.predict_layers[i](x.permute(0, 2, 1))  # (B, C, T) -> (B, C, pred_len)
            pred = pred.permute(0, 2, 1)  # (B, pred_len, C)
            # print("pred:", pred.shape)
            # 投影到最终输出通道数
            pred = self.projection_layer(pred)  # (B, pred_len, c_out)
            pred_list.append(pred)

        # 多尺度预测结果相加
        Y_pred = sum(pred_list)

        return Y_pred