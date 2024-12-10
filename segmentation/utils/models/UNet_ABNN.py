import torch
import torch.nn as nn

from utils.models.bnl import BNL
from utils.models.UNet import Upsampling

class ConvBlock(nn.Module):
    """
    Convolutional block with a bayesian normalization layer: 
        
        [Conv2d >> BNL >> ReLU] x 2

    Args:
        in_channels (int): number of input channels.
        hid_channels (int): number of channels in a hidden layer.
        out_channels (int): number of output channels.
        kernel_size (int or tuple): size of the convolving kernel.
        padding (int, tuple or str): padding added to all four sides of the input. 
    """
    def __init__(self, in_channels, hid_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.conv_block = nn.Sequential(*[nn.Conv2d(in_channels=in_channels, 
                                                    out_channels=hid_channels, 
                                                    kernel_size=kernel_size, 
                                                    padding=padding),
                                          BNL(hid_channels),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=hid_channels, 
                                                    out_channels=out_channels, 
                                                    kernel_size=kernel_size, 
                                                    padding=padding),
                                          BNL(out_channels),
                                          nn.ReLU()])
        
    def forward(self, x):
        return self.conv_block(x)
    
class UNet_Encoder(nn.Module):
    """
    Encoder part of the UNet with BNL blocks:

        [ConvBlock >> MaxPool2d] x 4 >> ConvBlock.

    Reduces image linear size in half for each MaxPool.
    Gradually increases the number of channels up until 1024.

    Args:
        in_channels (int): number of input channels.
    """
    def __init__(self, init_channels):
        super().__init__()

        self.conv0 = ConvBlock(in_channels=init_channels, hid_channels=64, out_channels=64)
        self.conv1 = ConvBlock(in_channels=64, hid_channels=128, out_channels=128)
        self.conv2 = ConvBlock(in_channels=128, hid_channels=256, out_channels=256)
        self.conv3 = ConvBlock(in_channels=256, hid_channels=512, out_channels=512)
        self.conv4 = ConvBlock(in_channels=512, hid_channels=1024, out_channels=1024)
        
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        e0 = self.conv0(x)
        e1 = self.conv1(self.pooling(e0))
        e2 = self.conv2(self.pooling(e1))
        e3 = self.conv3(self.pooling(e2))
        e4 = self.conv4(self.pooling(e3))

        encoder_outputs = [e0, e1, e2, e3]
        return e4, encoder_outputs
    
class UNet_Decoder(nn.Module):
    """
    Decoder part of the UNet with BNL blocks:

        [Upsampling >> ConvBlock] x 4

    After each Upsampling, the output is concatenated with an output of an encoder
    layer of the same depth. For more info, see https://arxiv.org/abs/1505.04597.

    Doubles image linear size for each Upsampling.
    Gradually decreases the number of channels up until the number of classes.

    Args:
        num_classes (int): number of classes to predict.
    """
    def __init__(self, num_classes):
        super().__init__()

        self.up0 = Upsampling(in_channels=1024, out_channels=512)
        self.up1 = Upsampling(in_channels=512, out_channels=256)
        self.up2 = Upsampling(in_channels=256, out_channels=128)
        self.up3 = Upsampling(in_channels=128, out_channels=64)

        self.deconv0 = ConvBlock(in_channels=1024, hid_channels=512, out_channels=512)
        self.deconv1 = ConvBlock(in_channels=512, hid_channels=256, out_channels=256)
        self.deconv2 = ConvBlock(in_channels=256, hid_channels=128, out_channels=128)
        self.deconv3 = ConvBlock(in_channels=128, hid_channels=64, out_channels=64)

        self.final = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x, encoder_outputs):
        d0 = self.up0(x)
        d0 = torch.cat([encoder_outputs[3], d0], dim = 1)
        d0 = self.deconv0(d0)

        d1 = self.up1(d0)
        d1 = torch.cat([encoder_outputs[2], d1], dim = 1)
        d1 = self.deconv1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([encoder_outputs[1], d2], dim = 1)
        d2 = self.deconv2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([encoder_outputs[0], d3], dim = 1)
        d3 = self.deconv3(d3)

        return self.final(d3)
    
class UNet_ABNN(nn.Module):
    """
    UNet architecture with BNL blocks consisting of a UNet encoder and UNet decoder
    with skip-connections.

    Args:
        in_channels (int): number of input channels.
        num_classes (int): number of classes to predict.
    """
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()

        self.encoder = UNet_Encoder(init_channels=n_channels)
        self.decoder = UNet_Decoder(num_classes=n_classes)
        
    def forward(self, x):
        x, encoder_outputs = self.encoder(x)
        return self.decoder(x, encoder_outputs)
    
if __name__ == "__main__":
    model = UNet_ABNN(init_channels=1)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))