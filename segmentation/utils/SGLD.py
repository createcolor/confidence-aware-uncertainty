"""
Taken from https://github.com/Unique-Divine/Langevin-Dynamics-for-NN-Optimization.
"""
import torch 
import numpy as np
from torch._C import Value
from torch.optim import optimizer
from typing import Generator, Iterable

Optimizer = optimizer.Optimizer # Base class for torch optimizers

class SGLD(Optimizer):
    """Stochastic Gradient Langevin Dynamics (SGLD).
    An algorithm for Bayesian learning from large scale datasets. 

    Weight decay is specified in terms of the Gaussian prior's sigma.

    Welling and Teh, 2011. Bayesian Learning via Stochastiv Gradient Langevin 
    Dynamics. Paper link: https://bit.ly/3ngnyRA

    Args:
        params (Iterable): an iterable of `torch.Tensor`s or
            `dict`s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
        lr (float): learning rate. 
        sigma_gauss_prior (float, optional): Defaults to 0.
        add_noise (bool, optional): Defaults to True. 
    
    Attributes:
        param_group (OptimizerParamGroup): Stores parameters in the param_group
            and stores a pointer to the OptimizerOptions. 
            docs: https://preview.tinyurl.com/2y272xmv

    Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of 
        dictionaries.
    """ 

    def __init__(self, params: Iterable, lr: float, 
                 sigma_gauss_prior: float = 100):
        if isinstance(sigma_gauss_prior, (complex)):
            if sigma_gauss_prior.imag != 0:
                raise ValueError(f"sigma_gauss_prior must be a real number.")

        weight_decay = 1 / (sigma_gauss_prior * sigma_gauss_prior)
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SGLD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """Updates neural network parameters. Called once the gradients are 
        computed using loss.backward(). Performs a single parameter update.

        Args:
            closure (callable, optional): A closure that reevaluates the 
                model and returns the loss. 
        This function should not modify the gradient field of the parameters, 
            i.e. `parameter.grad`. 
        """
        loss = None
        def params() -> Generator:
            for param_group in self.param_groups:
                weight_decay = param_group['weight_decay']
                for param in param_group['params']:
                    yield param, weight_decay, param_group
        
        # 'loss' gets updated from the following loop (under the hood)
        for param, weight_decay, param_group in params():
            if param.grad is None:
                continue
            gradient = param.grad.data
            if weight_decay != 0:
                gradient.add_(weight_decay, param.data)

            langevin_noise = param.data.new(param.data.size()).normal_(
                mean=0, std=1) * np.sqrt(param_group['lr'])
            param.data.add_(-param_group['lr'], 
                            0.5*gradient + langevin_noise)
            
            # if param_group['addnoise']:
            #     langevin_noise = param.data.new(param.data.size()).normal_(
            #         mean=0, std=1) / np.sqrt(param_group['lr'])
            #     param.data.add_(-param_group['lr'], 
            #                     0.5*gradient + langevin_noise)
            # else: # don't add noise
            #     param.data.add_(-param_group['lr'], 0.5*gradient)
        return loss

class PreconditionedSGLD(Optimizer):
    """Preconditioned Stochastic Gradient Langevin Dynamics (pSGLD).
    An algorithm that combines adaptive preconditioners (RMSprop) with SGLD. 

    Li, Chen, Carlson, and Carin, 2016. Preconditioned Stochastic Gradient 
        Langevin Dynamics for Deep Neural Networks. 
        Paper link: https://preview.tinyurl.com/25kd89a6
    Note, this is Algorithm 1 from the paper. 

    Weight decay is specified in terms of the Gaussian prior's sigma.

    Args:
        params (Iterable): an iterable of `torch.Tensor`s or
            `dict`s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
        lr (float): learning rate. 
        sigma_gauss_prior (float, optional): Defaults to 0.
        weight_balance (float, optional): Hyperparameter α (alpha) in the paper.
            Defaults to 0.99.
        step_size (float, optional): Hyperparameter ε_t in the paper. 
            Defaults to 1e-5.
        centered (bool, optional): Defautlts to False.
        add_noise (bool, optional): Defaults to True. 
    
    Attributes:
        param_group (OptimizerParamGroup): Stores parameters in the param_group
            and stores a pointer to the OptimizerOptions. 
            docs: https://preview.tinyurl.com/2y272xmv

    Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of 
        dictionaries.
    """ 

    def __init__(self, params: Iterable, lr: float,
                 sigma_gauss_prior: float = 0.1, weight_balance: float = 0.99, 
                 step_size: float = 1e-6, add_noise: bool =  True):
        if isinstance(sigma_gauss_prior, (complex)):
            if sigma_gauss_prior.imag != 0:
                raise ValueError(f"sigma_gauss_prior must be a real number.")

        weight_decay = 1 / (sigma_gauss_prior * sigma_gauss_prior)
        defaults = dict(
            lr=lr, weight_decay=weight_decay, step_size=step_size, 
            add_noise=add_noise)
        super(PreconditionedSGLD, self).__init__(params, defaults)

    def __setstate__(self, statue: dict) -> None:
        super(PreconditionedSGLD, self).__setstate__(statue)
        for param_group in self.param_groups:
            param_group.setdefault('centered', False)         

    def step(self, closure=None):
        """Updates neural network parameters. Called once the gradients are 
        computed using loss.backward(). Performs a single parameter update.

        Args:
            closure (callable, optional): A closure that reevaluates the 
                model and returns the loss. 
        This function should not modify the gradient field of the parameters, 
            i.e. `param.grad` in the code below. 
        """
        loss = None
        def params() -> Generator:
            for param_group in self.param_groups:
                weight_decay = param_group['weight_decay']
                for param in param_group['params']:
                    yield param, weight_decay, param_group
        
        # 'loss' gets updated from the following loop (under the hood)
        for param, weight_decay, param_group in params():
            if param.grad is None:
                continue
            gradient = param.grad.data
            state = self.state[param]
            state_is_empty: bool = len(state) == 0 
            if state_is_empty:
                state['step'] = 0
                state['sq_avg'] = torch.zeros_like(param.data)
                if param_group['centered']:
                    state['grad_avg'] = torch.zeros_like(param.data)

            sq_avg = state['sq_avg']
            weight_balance = param_group['weight_balance']
            state['step'] += 1

            if weight_decay != 0:
                gradient.add_(weight_decay, param.data)

            sq_avg.mul_(weight_balance).addcmul_(
                1 - weight_balance, gradient, gradient)
            
            if param_group['centered']:
                gradient_avg = state['gradient_avg']
                gradient_avg.mul_(weight_balance).add_(
                    1 - weight_balance, gradient)
                avg = sq_avg.cmul(-1, gradient_avg, gradient_avg).sqrt().add_(
                    param_group['step_size'])
            else:
                avg = sq_avg.sqrt().add_(param_group['step_size'])

            if param_group['addnoise']:
                langevin_noise = param.data.new(param.data.size()).normal_(
                    mean=0, std=1) / np.sqrt(param_group['lr'])
                param.data.add_(
                    -param_group['lr'], 
                    0.5*gradient.div_(avg) + (langevin_noise / torch.sqrt(avg)))
            else: # don't add noise
                param.data.add_(-param_group['lr'], 0.5*gradient, avg)

        return loss