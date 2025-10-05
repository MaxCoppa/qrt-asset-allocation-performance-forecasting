import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training and self.stddev > 0:
            return x + torch.randn_like(x) * self.stddev
        return x


class AE_BottleneckMLP(nn.Module):
    def __init__(self, num_columns, enc_units, dec_units, mlp_units, dropout_rate=0.2):
        """
        Args:
            num_columns (int): number of input features
            enc_units (list[int]): hidden sizes for encoder (e.g. [512, 256])
            dec_units (list[int]): hidden sizes for decoder (mirror of enc_units)
            mlp_units (list[int]): hidden sizes for supervised head (e.g. [128, 64])
            dropout_rate (float): dropout probability
        """
        super().__init__()

        # --- Input Normalization ---
        self.input_bn = nn.BatchNorm1d(num_columns)

        # --- Encoder ---
        enc_layers = []
        in_dim = num_columns
        enc_layers = [GaussianNoise(0.1)]
        in_dim = num_columns
        for u in enc_units:
            enc_layers += [
                nn.Linear(in_dim, u),
                nn.BatchNorm1d(u),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
            ]
            in_dim = u
        self.encoder = nn.Sequential(*enc_layers)

        # --- Decoder ---
        dec_layers = []
        in_dim = enc_units[-1]
        for u in dec_units:
            dec_layers += [
                nn.Linear(in_dim, u),
                nn.BatchNorm1d(u),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
            ]
            in_dim = u
        # final reconstruction to original dimension
        dec_layers += [nn.Linear(in_dim, num_columns)]
        self.decoder = nn.Sequential(*dec_layers)

        # --- Supervised MLP head (from bottleneck z) ---
        mlp_layers = []
        in_dim = enc_units[-1]
        for u in mlp_units:
            mlp_layers += [
                nn.Linear(in_dim, u),
                nn.BatchNorm1d(u),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
            ]
            in_dim = u
        mlp_layers += [nn.Linear(in_dim, 1)]  # output
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        x = self.input_bn(x)  # normalize input
        z = self.encoder(x)  # bottleneck representation
        x_recon = self.decoder(z)  # reconstruction
        y_pred = self.mlp(z)  # supervised output (logit/real)
        return x_recon, y_pred
