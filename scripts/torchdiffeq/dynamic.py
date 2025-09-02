import torch

# neural ode function
class ODEFunc(torch.nn.Module):
    def __init__(self, nNodes):
        
        super(ODEFunc, self).__init__()

        # define layers
        layers = []
        for i in range(len(nNodes)-1):
            layers.append(torch.nn.Linear(nNodes[i],nNodes[i+1],bias=True))
        self.layers = torch.nn.ModuleList(layers).double()

        # initialization
        for layer in self.layers:
            torch.nn.init.normal_(layer.weight, mean=0, std=0.1)
            torch.nn.init.constant_(layer.bias, val=0)

    def forward(self, t, y):
        if (len(y.size())==1):
            # reshape vector to row length 1 matrix
            y = torch.reshape(y,(1,-1)) 
        for i in range(len(self.layers)-1):
            y = torch.tanh(self.layers[i](y))
        y = self.layers[-1](y)
        return (y)