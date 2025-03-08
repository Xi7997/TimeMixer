# -*- coding: UTF-8 -*-
import torch
import torch.optim as optim
import torch.nn as nn
from model import TimeMixer
from Dataloader import get_dataloader
from tqdm import tqdm
# 训练超参数
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.01
SEQ_LEN = 96
PRED_LEN = 96
INPUT_DIM = 21

# 数据加载
train_loader = get_dataloader('../dataset/weather/weather.csv', batch_size=BATCH_SIZE, seq_len=SEQ_LEN, pred_len=PRED_LEN)

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = TimeMixer(input_dim=INPUT_DIM, seq_len=SEQ_LEN, pred_len=PRED_LEN).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# 训练循环
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()
        Y_pred = model(X)
        # print("shape:", Y_pred.shape, Y.shape)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")

# 保存模型
torch.save(model.state_dict(), "timemixer.pth")