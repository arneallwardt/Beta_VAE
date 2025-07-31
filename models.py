import torch
import torch.nn as nn

# this architecture follows the proposed architecture from the Factor VAE paper
class VAE_dsprites(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # [B, 1, 64, 64] -> [B, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),  # [B, 32, 32, 32] -> [B, 32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> [B, 64, 8, 8]
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),  # -> [B, 64, 4, 4]
            nn.ReLU(),
            nn.Flatten(),               # -> [B, 64*4*4 = 1024]
            nn.Linear(1024, 128),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 4 * 4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # -> [B, 64, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # -> [B, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # -> [B, 32, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),   # -> [B, 1, 64, 64]
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.decoder_input(z).view(-1, 64, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# this architecture follows the proposed architecture from the Factor VAE paper
class VAE_shapes(nn.Module):
    def __init__(self, latent_dim=6):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # [B, 3, 64, 64] -> [B, 32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),  # -> [B, 32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> [B, 64, 8, 8]
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),  # -> [B, 64, 4, 4]
            nn.ReLU(),
            nn.Flatten(),               # -> [B, 64*4*4 = 1024]
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 4 * 4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # -> [B, 64, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # -> [B, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # -> [B, 32, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),   # -> [B, 3, 64, 64]
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.decoder_input(z).view(-1, 64, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
