import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.nn import Linear
import numpy as np
from torch.nn.parameter import Parameter


class Encoder_Net(nn.Module):
    def __init__(self, layers, dims, adata, dropout):
        super(Encoder_Net, self).__init__()
        self.layers1 = nn.Linear(dims[0], dims[1])
        self.layers2 = nn.Linear(dims[0], dims[1])

        self.Decoder = decoder_ZINB(layers, dims[1], dims[0], dropout)

        self.adata = adata
        weights = self._initialize_weights()
        self.Coef = weights['Coef']

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['Coef'] = Parameter(1.0e-4 * torch.ones(size=(len(self.adata.X),len(self.adata.X))))
        return all_weights
    
    def forward(self, args ,x):
        mu = 0
        sigma = 1
        noise = np.random.normal(mu, sigma, size=x.shape)

        x1 = x.cpu().detach().numpy() + args.noise_lamda * noise
        x1 = torch.FloatTensor(x1)
        x1 = x1.to(args.device)
        out1 = self.layers1(x1)
        out2 = self.layers2(x)

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)

        z = (out1 + out2) / 2

        pi, disp, mean = self.Decoder(z)
        self.mean=mean

        return out1, out2, pi, disp, mean, self.Coef



