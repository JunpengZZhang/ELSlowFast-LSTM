import torch
import torch.nn as nn
import random

torch.manual_seed(0)
random.seed(0)

class BiLSTM(nn.Module):
    
    def __init__(self, input_size):
        super(BiLSTM, self).__init__()
        self.hidden_dim = 256
        self.num_layers = 2
        self.bilstm = nn.LSTM(input_size, self.hidden_dim // 2, num_layers=self.num_layers, dropout=0.75, bidirectional=True, bias=False)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, input_size)
        # LSTM 处理
        bilstm_out, _ = self.bilstm(x)  # 输出形状 (seq_len, batch_size, hidden_dim * 2)

        # 取最后一个时间步的输出作为特征
        last_time_step = bilstm_out[-1]  # 形状 (batch_size, hidden_dim * 2)

        return last_time_step  # 返回提取的特征


# input_size = 2048
# x = torch.randn(1, 8, input_size)  # (batch_size, seq_len, input_size)
# model = BiLSTM(input_size)
# out = model(x)
# print(out.shape)  # 输出形状应为 (1, C)，即 (1, 5)