import torch
from torch.nn import ModuleList, Linear, Tanh, Sequential, BatchNorm1d, ReLU, Dropout

'''
    Dynamic Module:
    This module is for setting up the ODE dynamic. 
'''

torch.set_default_dtype(torch.double)

class odeModule(torch.nn.Module):
    def __init__(self, fOde, theta):
        super().__init__()
        self.f = fOde
        if (not torch.is_tensor(theta)):
            theta = torch.tensor(theta)
        self.theta = torch.nn.Parameter(theta)

    def forward(self, x):
        if (x.ndimension()==1):
            x = x.unsqueeze(0)
        return (self.f(self.theta, x))

class nnMTModule(torch.nn.Module):
    # Multi-Task Neural Network
    def __init__(self, dim, hidden_nodes, dp=0):
        super().__init__()

        if not isinstance(hidden_nodes, list):
            hidden_nodes = [int(hidden_nodes)]

        # define network intermediate layers
        layers = ModuleList()
        nodes = [dim] + hidden_nodes + [dim]
        for i in range(1,len(nodes)-1):
            layers.append(
                Sequential(
                    Linear(nodes[i-1], nodes[i], bias=True),
                    # BatchNorm1d(nodes[i]),
                    ReLU(),
                    # Tanh(),
                    Dropout(p=dp)
                )
            )
        layers.append(
            Sequential(
                Linear(nodes[-2], nodes[-1], bias=True)
            )
        )
        self.networks = Sequential(*layers)

        # define output standardization
        self.dim = dim
        self.output_means = torch.zeros(dim)
        self.output_stds = torch.ones(dim)

    def update_output_layer(self, means, stds):
        self.output_means = means.detach()
        self.output_stds = stds.detach()

    def forward(self, x):
        if (x.ndimension()==1):
            x = x.unsqueeze(0)
        x = self.networks(x)
        for i in range(self.dim):
            x[:,i] = self.output_means[i] + self.output_stds[i] * x[:,i]
        return (x)


class nnSTModule(torch.nn.Module):
    # Single Task Neural Network
    def __init__(self, dim, hidden_nodes, dp=0):
        super().__init__()

        if not isinstance(hidden_nodes, list):
            hidden_nodes = [int(hidden_nodes)]
        
        # define network intermediate layers
        self.networks = ModuleList()
        for _ in range(dim):
            layers = ModuleList()
            nodes = [dim] + hidden_nodes + [1]
            for i in range(1,len(nodes)-1):
                layers.append(
                    Sequential(
                        Linear(nodes[i-1], nodes[i], bias=True),
                        # BatchNorm1d(nodes[i]),
                        ReLU(), # Tanh(),
                        Dropout(p=dp)
                    )
                )
            layers.append(
                Sequential(
                    Linear(nodes[-2], nodes[-1], bias=True)
                )
            )
            self.networks.append(Sequential(*layers))

        # define output standardization
        self.dim = dim
        self.output_means = torch.zeros(dim)
        self.output_stds = torch.ones(dim)

    def update_output_layer(self, means, stds):
        self.output_means = means.detach()
        self.output_stds = stds.detach()

    def forward(self, x):
        if (x.ndimension()==1):
            x = x.unsqueeze(0)
        x = torch.cat([self.networks[i](x) for i in range(self.dim)], 1)
        for i in range(self.dim):
            x[:,i] = self.output_means[i] + self.output_stds[i] * x[:,i]
        return (x)
