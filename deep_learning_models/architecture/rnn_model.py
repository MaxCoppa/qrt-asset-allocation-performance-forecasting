import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, seq_len=10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # add seq_len * input_dim extra inputs to the MLP
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 2 + seq_len * input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq, x_turnover, x_ret1):
        out, _ = self.lstm(x_seq)  # (batch, timesteps, hidden)
        out = out[:, -1, :]  # last hidden state (batch, hidden)
        seq_flat = x_seq.reshape(
            x_seq.size(0), -1
        )  # flatten (batch, seq_len * features)
        combined = torch.cat([out, seq_flat, x_turnover, x_ret1], dim=1)
        return self.fc(combined).squeeze()


class GRUModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim + 2, 1)

    def forward(self, x_seq, x_turnover, x_ret1):
        out, _ = self.gru(x_seq)
        out = out[:, -1, :]  # last hidden state
        combined = torch.cat([out, x_turnover, x_ret1], dim=1)
        return self.fc(combined).squeeze()
