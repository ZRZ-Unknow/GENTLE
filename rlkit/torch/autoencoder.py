import torch
import torch.nn as nn
from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp, ParallelLinear


class MlpEncoder(Mlp):
    def reset(self, num_tasks=1):
        pass

class MlpDecoder(PyTorchModule):
    def __init__(self,
                 hidden_size,
                 num_hidden_layers,
                 z_dim,
                 action_dim,
                 obs_dim,
                 reward_dim,
                 use_next_obs_in_context,
                 ensemble_size=1):
        super().__init__()
        self.action_dim=action_dim
        self.obs_dim=obs_dim
        self.reward_dim=reward_dim
        self.z_dim = z_dim
        self.use_next_obs_in_context = use_next_obs_in_context
        self.ensemble_size = ensemble_size

        if ensemble_size == 1: 
            self.backbones = Mlp(input_size=z_dim + obs_dim + action_dim,
                           output_size=1+obs_dim if use_next_obs_in_context else 1,
                           hidden_sizes=[hidden_size for i in range(num_hidden_layers)]) 
        else:
            self.backbones = []
            for i in range(num_hidden_layers):
                if i == 0:
                    self.backbones.append(ParallelLinear(obs_dim + action_dim + z_dim, hidden_size, ensemble_size))
                else:
                    self.backbones.append(ParallelLinear(hidden_size, hidden_size, ensemble_size))
                self.backbones.append(nn.ReLU())
            self.backbones.append(ParallelLinear(hidden_size, 1+obs_dim if use_next_obs_in_context else 1, ensemble_size))
            self.backbones = nn.Sequential(*self.backbones)

    def forward(self, obs, action, z, return_std=False):
        inputs = torch.cat([obs, action, z], -1)
        if len(inputs.shape) == 3:
            inputs = torch.flatten(inputs, start_dim=0, end_dim=1)
        outputs = self.backbones(inputs)
        if self.ensemble_size > 1:
            stds = torch.std(outputs, dim=0)
            outputs = torch.mean(outputs, dim=0)
        if outputs.shape[:-1] != obs.shape[:-1]:
            outputs = outputs.reshape((obs.shape[0], obs.shape[1], -1))
        if self.use_next_obs_in_context:
            reward, next_obs = torch.split(outputs, [1, self.obs_dim], dim=-1)
            next_obs = next_obs + obs
            outputs = torch.cat([reward, next_obs], dim=-1)
        if return_std:
            return outputs, stds
        return outputs

