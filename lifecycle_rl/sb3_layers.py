'''
SB3-verkkokerrokset lifecycle-mallin RL-ajoihin
'''

from typing import Sequence, Type

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LayerNormMlpExtractor(BaseFeaturesExtractor):
    '''
    MLP-runko, jossa jokainen piilokerros on Linear -> LayerNorm -> aktivointi,
    kuten SB2:n LnMlpPolicy.

    ACKTR:n K-FAC esikunnittaa vain Linear-kerrokset; LayerNormin skaala- ja
    siirtoparametrit päivittyvät K-FAC:n euklidista momentum-polkua.
    '''

    def __init__(
        self,
        observation_space: spaces.Space,
        net_arch: Sequence[int] = (256, 256, 64),
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        elementwise_affine: bool = True,
    ):
        if len(net_arch) < 1:
            raise ValueError('net_arch must contain at least one hidden layer.')

        layers: list[nn.Module] = [nn.Flatten()]
        last_dim = get_flattened_obs_dim(observation_space)
        for width in net_arch:
            width = int(width)
            layers.append(nn.Linear(last_dim, width))
            layers.append(nn.LayerNorm(width, elementwise_affine=elementwise_affine))
            layers.append(activation_fn())
            last_dim = width

        super().__init__(observation_space, features_dim=last_dim)
        self.net = nn.Sequential(*layers)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)
