import torch
import torch.nn as nn
import numpy as np


# Helper function to construct a feedforward multilayer perception that you can use in class Net if you want.
# Do NOT modify, as this is used to construct the policy network.
# Sizes is a list of the number of neurons in each layer (where the number of layers is the len of the list)
# nn.Sequential allows you to skip explicitly defining a forward pass like you had to do in assignment 1.
# See pyTorch documentation if you're confused (or feel free to define the network and forward() by hand as in assignment 1).
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    # Build a feedforward neural network for the policy
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


class Net(nn.Module):
    """Reward model: scalar reward per state; trajectory return is sum over timesteps."""

    def __init__(self):
        super().__init__()
        self.net = mlp(sizes=[4, 32, 32, 1], activation=nn.ReLU)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)

    def predict_return(self, traj):
        '''calculate return (cumulative reward) of a trajectory (could be any number of timesteps)'''
        if not isinstance(traj, torch.Tensor):
            traj = torch.as_tensor(traj, dtype=torch.float32, device=next(self.parameters()).device)
        else:
            traj = traj.float().to(next(self.parameters()).device)
        if traj.dim() == 1:
            traj = traj.unsqueeze(0)
        return self.forward(traj).sum()
