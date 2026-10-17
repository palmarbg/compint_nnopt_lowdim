import torch.nn as nn
from torch.func import functional_call
import torch

class VectorMLP(nn.Module):
    def __init__(self, model, lower_dim=500):
        super().__init__()

        self.model = model
        self.lower_dim = lower_dim
        
        self.num_weights = 0
        for _, param in model.named_parameters():
            param.requires_grad_(False)
            self.num_weights += param.numel()

        # Non trainable parameters
        self.register_buffer(
            "proj_matrix",
            torch.randn((self.num_weights, self.lower_dim))
        )
        self.register_buffer(
            "bias",
            torch.randn((self.num_weights, )) / (self.num_weights ** 0.5)
        )

        # Trainable parameters
        self.lower_weights = nn.Parameter(torch.zeros((lower_dim,), requires_grad=True))

    def forward(self, x):
        higher_weights = self.bias + self.proj_matrix @ self.lower_weights

        params = {}
        i = 0

        for name, param in self.model.named_parameters():
            params[name] = higher_weights[i:i+param.numel()].view(param.shape)
            i += param.numel()

        return functional_call(self.model, params, (x,))

    def train(self, mode = True):
        self.model.train()
        return super().train(mode)

    def eval(self):
        self.model.eval()
        return super().eval()

    def zero_grad(self, set_to_none = True):
        self.model.zero_grad(set_to_none)
        return super().zero_grad(set_to_none)

    def parallel_eval(self, weights, x):
        higher_weights = self.bias + self.proj_matrix @ weights

        params = {}
        i = 0

        for name, param in self.model.named_parameters():
            params[name] = higher_weights[i:i+param.numel()].view(param.shape)
            i += param.numel()

        return functional_call(self.model, params, (x,))

