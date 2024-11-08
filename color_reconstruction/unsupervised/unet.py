import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def create_positional_encoding(max_t=1000, t_dim=128):
    pe = torch.zeros(max_t, t_dim)
    position = torch.arange(0, max_t, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, t_dim, 2).float() * (-math.log(10000.0) / t_dim)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


class DownBlock(torch.nn.Module):
    def __init__(self, in_size, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels * 2,
            kernel_size=3,
            padding=1,
        )
        self.norm1 = torch.nn.BatchNorm2d(num_channels)
        self.norm2 = torch.nn.BatchNorm2d(num_channels * 2)
        self.activation = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # print(    f"DownBlock :  In size : {x.shape}" )
        # First convolution block
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)

        # Second convolution block
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)

        # Apply pooling for the next layer
        h = self.pool(h)
        # print(f"DpwnBlock : out_size : {h.shape}")
        return h, h


class UpBlock(torch.nn.Module):
    def __init__(self, in_size, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.upsample = torch.nn.ConvTranspose2d(
            in_channels=num_channels * 2,
            out_channels=num_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,
        )
        self.norm1 = torch.nn.BatchNorm2d(num_channels)
        self.norm2 = torch.nn.BatchNorm2d(num_channels)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        # print(f"UpBlock :  In size : {x.shape}, {self.num_channels}")
        # Upsample
        h = self.upsample(x)

        # First convolution block
        h = self.conv1(h)
        h = self.norm1(h)
        h = self.activation(h)

        # Second convolution block
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        # print(f"UpBlock :  Out size : {h.shape}")
        return h


class Unet(torch.nn.Module):

    def __init__(self, img_size=256, channels=128, latent_space_dim=None):
        super().__init__()
        if latent_space_dim == None:
            latent_space_dim = img_size // 8
        self.latent_space_dim = latent_space_dim
        self.pe = create_positional_encoding().to("cuda")
        self.num_blocks = int(np.log2(img_size // latent_space_dim))

        # Add initial convolution to handle input channels
        self.initial_conv = torch.nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.linears_down = torch.nn.ModuleList(
            [
                torch.nn.Linear(128, int(channels * math.pow(2, i)))
                for i in range(self.num_blocks)
            ]
        )
        self.linear_first = torch.nn.Linear(128, 128)
        self.linears_up = torch.nn.ModuleList(
            [
                torch.nn.Linear(128, int(channels * math.pow(2, i) * 2))
                for i in range(self.num_blocks)
            ]
        )
        self.down_blocks = torch.nn.ModuleList(
            [
                DownBlock(
                    in_size=img_size // math.pow(2, i),
                    num_channels=int(channels * math.pow(2, i)),
                )
                for i in range(self.num_blocks)
            ]
        )
        self.up_blocks = torch.nn.ModuleList(
            [
                UpBlock(
                    in_size=img_size // math.pow(2, i),
                    num_channels=int(channels * math.pow(2, i)),
                )
                for i in range(self.num_blocks)
            ]
        )

        # Add final convolution to map back to 3 channels
        self.final_conv = torch.nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x_t, Ts):
        # Initial convolution
        x_t = self.initial_conv(x_t)
        Ts_embed = self.pe[
            Ts.view(
                -1,
            )
        ]
        Ts_embed = torch.nn.functional.relu(self.linear_first(Ts_embed))
        res = []
        # encoding
        for i in range(self.num_blocks):
            x_t = x_t + torch.nn.functional.relu(self.linears_down[i](Ts_embed)).view(
                x_t.shape[0], x_t.shape[1], 1, 1
            )
            x_t, s_i = self.down_blocks[i](x_t)
            res.append(s_i)

        # don't put last skip
        res[-1] = torch.zeros_like(x_t)
        # decoding
        for i in range(self.num_blocks):
            x_t = x_t + torch.nn.functional.relu(
                self.linears_up[self.num_blocks - 1 - i](Ts_embed)
            ).view(x_t.shape[0], x_t.shape[1], 1, 1)
            x_t = self.up_blocks[self.num_blocks - 1 - i](
                x_t + res[self.num_blocks - 1 - i]
            )

        # Final convolution to get back to image channels
        x_t = self.final_conv(x_t)

        return x_t





class SmallUNet(nn.Module):
    def __init__(self, input_channels=3, base_channels=64):
        super(SmallUNet, self).__init__()

        # Encoder layers
        self.encoder1 = nn.Sequential(nn.Conv2d(input_channels, base_channels, 3, padding=1), nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Conv2d(base_channels, base_channels * 2, 3, padding=1), nn.ReLU())

        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1), nn.ReLU())

         # Decoder layers
        self.decoder1 = nn.Sequential(nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1), nn.ReLU())
        self.decoder2 = nn.Sequential(nn.Conv2d(base_channels * 2, base_channels, 3, padding=1), nn.ReLU())

        # Output layer
        self.output = nn.Conv2d(base_channels, input_channels, 1)

    def forward(self, x, t):
        # Encode
        x1 = self.encoder1(x)
        x2 = self.encoder2(F.max_pool2d(x1, 2))

        # Bottleneck
        x_bottleneck = self.bottleneck(F.max_pool2d(x2, 2))

        # Decode
        x = F.interpolate(self.decoder1(x_bottleneck), scale_factor=2)
        x = F.interpolate(self.decoder2(x + x2), scale_factor=2)

        # Output layer
        return self.output(x + x1)




if __name__ == "__main__":
    model = Unet()
    print(sum(p.numel() for p in model.parameters()))
    test_input = torch.zeros((1, 3, 256, 256))
    output = model(test_input)
    print(output.shape)