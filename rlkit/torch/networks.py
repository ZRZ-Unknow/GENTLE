"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
import numpy as np
from torch import nn as nn
from torch.nn import functional as F
from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm
from typing import Dict, List, Union, Tuple, Optional
# import rlkit.torch.transformer as transformer

def identity(x):
    return x

class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            batch_attention=False,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            use_dropout=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.use_dropout = use_dropout
        self.fcs = []
        self.layer_norms = []
        self.dropouts = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)
            
            if self.use_dropout:
                dropout_n = nn.Dropout(0.1)
                self.__setattr__("drop_out{}".format(i), dropout_n)
                self.dropouts.append(dropout_n)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)
        
        self.batch_attention = batch_attention
        if self.batch_attention:
            self.attn_pooling = nn.MultiheadAttention(embed_dim=output_size, num_heads=1, batch_first=True)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
            if self.use_dropout and i < len(self.fcs) - 1:
                h = self.dropouts[i](h)
        preactivation = self.last_fc(h)
        if self.batch_attention:
            output = self.attn_pooling(torch.mean(preactivation, dim=1, keepdim=True), preactivation, preactivation)[0]
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, meta_size=16, batch_size=256, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)

class FlattenMlpDecoder(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)

class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)



class ParallelLinear(PyTorchModule):
    def __init__(self, in_features, out_features, ensemble_size=7):
        super().__init__()
        self.ensemble_size = ensemble_size
        
        self.weight = torch.zeros(ensemble_size, in_features, out_features)
        self.bias = torch.zeros(ensemble_size, 1, out_features)
        self.weight = torch.nn.Parameter(self.weight)
        self.bias = torch.nn.Parameter(self.bias)
        
        torch.nn.init.trunc_normal_(self.weight, std=1/(2*in_features**0.5))
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, self.weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, self.weight)
        x += self.bias
        
        return x


class EnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        weight_decay: float = 0.0
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble
        self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))
        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))
        self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias
        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)
        x = x + bias
        return x

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.weight_decay * (0.5*((self.weight**2).sum()))
        return decay_loss


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x


def soft_clamp(
    x : torch.Tensor,
    _min: Optional[torch.Tensor] = None,
    _max: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        num_elites: int = 5,
        activation: nn.Module = Swish,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        with_next_obs: bool = False,
    ) -> None:
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_next_obs = with_next_obs
        self.device = ptu.device

        self.activation = activation()

        assert len(weight_decays) == (len(hidden_dims) + 1)

        module_list = []
        hidden_dims = [obs_dim+action_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        for in_dim, out_dim, weight_decay in zip(hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1]):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
        self.backbones = nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(
            hidden_dims[-1],
            (1+obs_dim) if with_next_obs else 1,
            num_ensemble,
            weight_decays[-1]
        )
        self.to(self.device)

    def forward(self, obs_act) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_act = torch.as_tensor(obs_act, dtype=torch.float32).to(self.device)
        output = obs_act
        for layer in self.backbones:
            output = self.activation(layer(output))
        output = self.output_layer(output)
        if self._with_next_obs:
            reward, next_obs = torch.split(output, [1, self.obs_dim], dim=-1)
            obs, act =  torch.split(obs_act, [self.obs_dim, self.action_dim], dim=-1)
            next_obs = next_obs + obs
            output = torch.cat([reward, next_obs], dim=-1)
        return output

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss

    def random_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.num_elites, size=batch_size)
        return idxs