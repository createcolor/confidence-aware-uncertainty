"""
Adopted from https://github.com/mobarakol/SVLS.
"""
import torch
import math


def get_gaussian_kernel_2d(ksize=0, sigma=0):
    x_grid = torch.arange(ksize).repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (ksize - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance + 1e-16)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance + 1e-16)
        )
    return gaussian_kernel / torch.sum(gaussian_kernel)


class get_svls_filter_2d(torch.nn.Module):
    def __init__(self, ksize=3, sigma=0, channels=0):
        super(get_svls_filter_2d, self).__init__()

        gkernel = get_gaussian_kernel_2d(ksize=ksize, sigma=sigma)

        neighbors_sum = (1 - gkernel[1, 1]) + 1e-16
        gkernel[int(ksize / 2), int(ksize / 2)] = neighbors_sum

        self.svls_kernel = gkernel / neighbors_sum
        svls_kernel_2d = self.svls_kernel.view(1, 1, ksize, ksize)
        svls_kernel_2d = svls_kernel_2d.repeat(channels, 1, 1, 1)
        padding = int(ksize / 2)

        self.svls_layer = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                          kernel_size=ksize, groups=channels,
                                          bias=False, padding=padding, padding_mode='replicate')
        self.svls_layer.weight.data = svls_kernel_2d
        self.svls_layer.weight.requires_grad = False

    def forward(self, x):
        return self.svls_layer(x) / self.svls_kernel.sum()


class CELossWithSVLS_2D(torch.nn.Module):
    def __init__(self, classes=None, sigma=1, ksize=3, device='cpu'):
        super(CELossWithSVLS_2D, self).__init__()
        self.cls = torch.tensor(classes)
        self.svls_layer = get_svls_filter_2d(ksize=ksize, sigma=sigma, channels=self.cls).to(device)

    def forward(self, inputs, labels, eps=1e-8):
        labels = torch.cat((labels, 1. - labels), dim=1)
        svls_labels = self.svls_layer(labels)
        inputs = torch.log(torch.cat((inputs, 1. - inputs), dim=1) + eps)
        return (-svls_labels * inputs).sum(dim=1).mean()


class CELossWithOH_2D(torch.nn.Module):
    def __init__(self, classes=None):
        super(CELossWithOH_2D, self).__init__()
        self.cls = torch.tensor(classes)

    def forward(self, inputs, labels, eps=1e-8):
        labels = torch.cat((labels, 1. - labels), dim=1)
        inputs = torch.log(torch.cat((inputs, 1. - inputs), dim=1) + eps)
        return (-labels * inputs).sum(dim=1).mean()
