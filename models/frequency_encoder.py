import torch.nn as nn


class FrequencyEncoderMAE(nn.Module):
    """
    Simple masked-autoencoder for frequency vectors
    """

    def __init__(self, n_freqs, latent_dim, hidden=512, proj_dim=128):
        super().__init__()
        self.n_freqs = n_freqs
        self.encoder = nn.Sequential(
            nn.Linear(n_freqs, hidden), nn.GELU(), nn.Linear(hidden, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, n_freqs)
        )
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x_masked):
        z = self.encoder(x_masked)
        recon = self.decoder(z)
        proj = self.proj(z)
        return recon, z, proj
