"""

This file defines the key components of the Adaptive Bayesian Neural Network (ABNN) model, 
including the Bayesian Normalization Layer (BNL) and the custom Maximum A Posteriori (MAP) 
loss function. These components are essential in the implementation of the ABNN as described
in the paper "Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from 
Pre-trained Models" (https://arxiv.org/abs/2312.15297).

"""

import torch
import torch.nn as nn


class CustomMAPLoss(nn.Module):
    def __init__(self, eta, model_parameters, prior_std=1.0):
        super(CustomMAPLoss, self).__init__()
        self.eta = eta  # Class-dependent random weights
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.prior_std = prior_std
        self.model_parameters = model_parameters

    def forward(self, outputs, targets):
        device = outputs.device  # Get the device of the outputs tensor
        targets = targets.to(device)  # Move targets to the same device

        # Ensure eta is on the same device
        self.eta = self.eta.to(device)

        # Cross-entropy loss (negative log-likelihood)
        nll_loss = self.cross_entropy(outputs, targets)

        # Perturbation term: E(ω) = -∑ ηi logP(yi | xi, ω)
        perturbation_loss = self.eta[targets] * nll_loss

        # MAP loss: LMAP(ω) = -∑ logP(yi | xi, ω) - logP(ω)
        map_loss = nll_loss.mean() + self.prior_log_prob(self.model_parameters, self.prior_std)

        # Total loss: L(ω) = LMAP(ω) + E(ω)
        total_loss = map_loss + perturbation_loss.mean()

        return total_loss



    @staticmethod
    def prior_log_prob(params, std):
        # Compute logP(ω) for a normal prior with standard deviation `std`
        log_prob = 0.0
        for param in params:
            param = param.to(std)  # Move param to the same device as std
            log_prob += -0.5 * torch.sum(param ** 2) / (std ** 2)
        return log_prob

class BinaryABNNLoss(torch.nn.Module):
    def __init__(self, Num_classes, model_parameters, Weight_decay=1e-4):
        super(BinaryABNNLoss, self).__init__()
        self.model_parameters = model_parameters
        self.Weight_decay = Weight_decay
        self.eta = nn.Parameter(torch.ones((1,)))

    def forward(self, outputs, labels):
        # Calculate the three loss components
        nll_loss = self.negative_log_likelihood(outputs, labels)
        log_prior_loss = self.negative_log_prior(self.model_parameters, self.Weight_decay)
        custom_ce_loss = self.custom_cross_entropy_loss(outputs, labels, self.eta)

        # Sum up all three components to form the ABNN loss
        total_loss = nll_loss + log_prior_loss + custom_ce_loss
        # print(nll_loss, log_prior_loss, custom_ce_loss)
        return total_loss

    @staticmethod
    def negative_log_likelihood(outputs, labels):
        # Negative Log Likelihood (NLL) or MLE Loss:
        # NLL = -∑ log P(y_i | x_i, ω)
        return torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels)

    def negative_log_prior(self, model_parameters, Weight_decay=1e-4):
        # Negative Log Prior with Gaussian Prior (L2 Regularization):
        # log P(ω) = λ ∑ ω^2 where λ (weight decay) = (1/2σ^2)
        l2_reg = sum(p.pow(2).sum() for p in model_parameters)
        return Weight_decay * l2_reg

    def custom_cross_entropy_loss(self, outputs, labels, eta):
        # Custom Cross-Entropy Loss:
        # E(ω) = -∑ η_i log P(y_i | x_i, ω)
        # log_probs = torch.log(torch.sigmoid(outputs, dim=1) + 1e-8)

        inputs = torch.sigmoid(outputs)
        log_probs = torch.log(torch.cat((inputs, 1 - inputs), dim=1) + 1e-8)
        # weighted_log_probs = eta[labels] * log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        labels = torch.cat((labels, 1. - labels), dim=1)
        labels = labels == 1.0
        # print(eta.shape, labels.sum())
        # print(eta[labels].shape, eta[labels].sum())
        weighted_log_probs = eta * labels * log_probs
        # print(-torch.mean(weighted_log_probs))
        return -torch.mean(weighted_log_probs)

class ABNNLoss(torch.nn.Module):
    def __init__(self, Num_classes, model_parameters, Weight_decay=1e-4):
        super(ABNNLoss, self).__init__()
        self.model_parameters = model_parameters
        self.Weight_decay = Weight_decay
        self.eta = nn.Parameter(torch.ones(Num_classes))

    def forward(self, outputs, labels):
        # Calculate the three loss components
        nll_loss = self.negative_log_likelihood(outputs, labels)
        log_prior_loss = self.negative_log_prior(self.model_parameters, self.Weight_decay)
        custom_ce_loss = self.custom_cross_entropy_loss(outputs, labels, self.eta)

        # Sum up all three components to form the ABNN loss
        total_loss = nll_loss + log_prior_loss + custom_ce_loss
        return total_loss

    @staticmethod
    def negative_log_likelihood(outputs, labels):
        # Negative Log Likelihood (NLL) or MLE Loss:
        # NLL = -∑ log P(y_i | x_i, ω)
        return torch.nn.functional.cross_entropy(outputs, labels)

    def negative_log_prior(self, model_parameters, Weight_decay=1e-4):
        # Negative Log Prior with Gaussian Prior (L2 Regularization):
        # log P(ω) = λ ∑ ω^2 where λ (weight decay) = (1/2σ^2)
        l2_reg = sum(p.pow(2).sum() for p in model_parameters)
        return Weight_decay * l2_reg

    def custom_cross_entropy_loss(self, outputs, labels, eta):
        # Custom Cross-Entropy Loss:
        # E(ω) = -∑ η_i log P(y_i | x_i, ω)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        weighted_log_probs = eta[labels] * log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        return -torch.mean(weighted_log_probs)
