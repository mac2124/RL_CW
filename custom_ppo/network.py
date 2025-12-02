import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64], activation_fn=F.relu):
        super(FeedForwardNN, self).__init__()
        self.activation_fn = activation_fn

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.output_layer = nn.Linear(hidden_dims[1], output_dim)


    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        
        # Ensure the input tensor has the correct shape
        if x.ndim > 2:
            x = x.reshape(x.size(0), -1)  # Flatten the input if it has more than 2 dimensions
        elif x.ndim == 1:
            x = x.unsqueeze(0)  # Add batch dimension if input is a single vector
        
        activation_input = self.activation_fn(self.input_layer(x))
        activation_hidden = self.activation_fn(self.layer2(activation_input))
        output = self.output_layer(activation_hidden)
        
        return output