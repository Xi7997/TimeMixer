import torch
import numpy as np
from model import TimeMixer
from Dataloader import get_dataloader

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = TimeMixer()
model.load_state_dict(torch.load("timemixer.pth"))
model.to(device)
model.eval()

# 加载测试数据
test_loader = get_dataloader('../dataset/weather/weather.csv', batch_size=32, seq_len=96, pred_len=96)

# 评估
mse_losses = []
mae_losses = []
criterion_mse = torch.nn.MSELoss()
criterion_mae = torch.nn.L1Loss()

with torch.no_grad():
    for X, Y in test_loader:
        X, Y = X.to(device), Y.to(device)
        Y_pred = model(X)

        mse_loss = criterion_mse(Y_pred, Y).item()
        mae_loss = criterion_mae(Y_pred, Y).item()

        mse_losses.append(mse_loss)
        mae_losses.append(mae_loss)

print(f"Test MSE: {np.mean(mse_losses):.4f}")
print(f"Test MAE: {np.mean(mae_losses):.4f}")
