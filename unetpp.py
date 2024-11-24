import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Two convolutional layers followed by BatchNorm and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNetPlusPlus, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.dense_blocks = nn.ModuleDict()

        # Encoder: Down-sampling path
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder: Up-sampling path
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc_features = []  # To store features from each encoder level
        for encode in self.encoder:
            x = encode(x)
            enc_features.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Decoder path with nested skip connections
        x = enc_features[-1]  # Start from the bottleneck feature map
        for i in range(len(self.decoder)):
            x = self.upconvs[i](x)  # Upsample
            skip_connection = enc_features[-(i + 2)]  # Corresponding encoder layer
            x = torch.cat((skip_connection, x), dim=1)  # Concatenate skip connection
            x = self.decoder[i](x)  # Decode

        # Final output
        return self.final_conv(x)