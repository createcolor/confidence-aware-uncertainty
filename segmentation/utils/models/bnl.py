import torch
import torch.nn as nn


class BNL(nn.Module):
    """
        Bayesian Normalization Layer (BNL).

    This layer replaces traditional normalization layers like BatchNorm,
    LayerNorm, and InstanceNorm. It adapts normalization to account for Bayesian
    inference, making the model more robust to variations and uncertainties in
    the data.

    BNL adds gaussian noise during both inference and trainig stages.

    This implementation includes parameters named `weight` and `bias` to directly
    match those used in PyTorch's BatchNorm, LayerNorm, and InstanceNorm layers 
    for compatibility when loading state dictionaries.

    Args:
        num_features (int, list, tuple): Number of features in the input, matches channels 
                                         in conv layers or features in linear layers. Can
                                         be a single integer or a list/tuple for complex scenarios.
    """
    def __init__(self, num_features):
        super(BNL, self).__init__()
        # Check if num_features is a list or tuple, convert if necessary
        if isinstance(num_features, int):
            num_features = (num_features,)
        
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = 1e-5

    def forward(self, x):
        if len(self.num_features) == 1:  # Traditional usage like BatchNorm
            mean = x.mean([0, 2, 3], keepdim=True) if x.dim() == 4 else x.mean(0, keepdim=True)
            var = x.var([0, 2, 3], keepdim=True) if x.dim() == 4 else x.var(0, keepdim=True)
            x_normalized = (x - mean) / torch.sqrt(var + self.eps)

            noise = torch.randn(self.weight.shape, device=x.device)
            gamma_noisy = self.weight * (1 + noise)

            if x.dim() == 4:
                gamma_noisy = gamma_noisy.view(1, -1, 1, 1)
                bias = self.bias.view(1, -1, 1, 1)
            elif x.dim() == 2:
                gamma_noisy = gamma_noisy.view(1, -1)
                bias = self.bias.view(1, -1)

            return gamma_noisy * x_normalized + bias
        else:  # LayerNorm-like usage
            mean = x.mean(dim=tuple(range(x.dim())[1:]), keepdim=True)
            var = x.var(dim=tuple(range(x.dim())[1:]), keepdim=True, unbiased=False)
            x_normalized = (x - mean) / torch.sqrt(var + self.eps)

            noise = torch.randn(self.weight.shape, device=x.device)
            gamma_noisy = self.weight * (1 + noise)

            weight = self.weight.view((1,) + self.num_features + (1,) * (x.dim() - len(self.num_features) - 1))
            bias = self.bias.view((1,) + self.num_features + (1,) * (x.dim() - len(self.num_features) - 1))

            return gamma_noisy * x_normalized + bias
