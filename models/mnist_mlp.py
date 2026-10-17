import torch.nn as nn

class MNIST_MLP(nn.Module):
    def __init__(self, hidden_dims=[]):
        super().__init__()

        layer_dims = [784, *hidden_dims, 10]
        n = len(layer_dims)

        self.model = nn.Sequential()
        for i in range(n-2):
            self.model.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            self.model.append(nn.ReLU())

        self.model.append(nn.Linear(layer_dims[n-2], layer_dims[n-1]))

    def forward(self, x):
        return self.model(x)