import torch
import torch.nn as nn

class HashToGaussianGenerator(nn.Module):
    def __init__(self, hash_dim=64, noise_dim=4*64*64):
        super(HashToGaussianGenerator, self).__init__()
        self.hash_dim = hash_dim
        self.noise_dim = noise_dim

        # 编码器：hash to noise
        self.encoder = nn.Sequential(
            self.make_conv_layer(1, 32),
            self.make_conv_layer(32, 128),
            self.make_conv_layer(128, 512),
            self.make_conv_layer(512, 128),
            nn.Flatten(),
            nn.Linear(128 * self.hash_dim, self.noise_dim),
        )

        # 解码器：noise to hash
        self.decoder = nn.Sequential(
            self.make_linear_layer(self.noise_dim, 128 * self.hash_dim),
            nn.Unflatten(1, (128, self.hash_dim)),
            self.make_convT_layer(128, 32),
            nn.ConvTranspose1d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, rev=False):
        if not rev:
            x = x.unsqueeze(1)  # 调整形状[batch_size, 1, hash_dim]
            h = self.encoder(x)
            x2 = self.decoder(h).squeeze(1)

            return h, x2
        else:
            x1 = x
            x2 = self.decoder(x1).squeeze(1) # noise → hash

            return x1, x2

    @staticmethod
    def make_conv_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

    @staticmethod
    def make_convT_layer(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

    @staticmethod
    def make_linear_layer(in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.BatchNorm1d(out_features)
        )
