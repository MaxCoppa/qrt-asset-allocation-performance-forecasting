import torch
import torch.nn as nn


class CrossSeriesModel(nn.Module):
    def __init__(self, seq_dim=2, hidden_dim=64, num_heads=4, dropout=0.2):
        super().__init__()

        # Encode each TS with LSTM
        self.encoder = nn.LSTM(
            input_size=seq_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Cross-series self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # Projection for raw x_seq → hidden space
        self.proj_raw = nn.Linear(seq_dim, hidden_dim)

        # Final prediction head
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),  # raw + seq_emb + attn
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        x: (B, N, T, F)
        B = batch size
        N = number of TS
        T = timesteps
        F = input features
        """
        B, N, T, F = x.shape

        # Encode each TS independently with LSTM
        x_flat = x.reshape(B * N, T, F)
        seq_out, _ = self.encoder(x_flat)
        seq_emb = seq_out[:, -1, :]  # (B*N, hidden_dim)
        seq_emb = seq_emb.reshape(B, N, -1)  # (B, N, hidden_dim)

        # Cross-series self-attention
        attn_out, _ = self.attn(seq_emb, seq_emb, seq_emb)

        # Residual connection (like Transformer: seq_emb + attn_out)
        seq_plus_attn = seq_emb + attn_out  # (B, N, hidden_dim)

        # Raw input features → hidden space (use last timestep as summary)
        raw_last = x[:, :, -1, :]  # (B, N, F)
        raw_emb = self.proj_raw(raw_last)  # (B, N, hidden_dim)

        # Concatenate raw info + seq_emb + attention
        h = torch.cat([raw_emb, seq_emb, seq_plus_attn], dim=-1)  # (B, N, 3*hidden_dim)

        # Per-TS prediction
        return self.fc_out(h).squeeze(-1)  # (B, N)
