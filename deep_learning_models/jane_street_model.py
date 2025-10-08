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
    def __init__(
        self, num_columns, enc_units, dec_units, mlp_units, recon_dim, dropout_rate=0.2
    ):
        super().__init__()
        self.recon_dim = recon_dim

        # --- Input Normalization ---
        self.input_bn = nn.BatchNorm1d(num_columns)

        # --- Encoder ---
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

        # --- Decoder (reconstruction only target vector) ---
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
        # final output = recon_dim
        dec_layers += [nn.Linear(in_dim, recon_dim)]
        self.decoder = nn.Sequential(*dec_layers)

        # --- Supervised MLP head ---
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
        mlp_layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        x = self.input_bn(x)
        z = self.encoder(x)
        x_recon = self.decoder(z)  # reconstruit uniquement le vecteur cible
        y_pred = self.mlp(z)
        return x_recon, y_pred
