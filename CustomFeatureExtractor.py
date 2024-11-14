import gymnasium as gym
import torch as th
from torch import nn
import numpy as np

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        

        for key, subspace in observation_space.spaces.items():
            if key == "laser":
                n_laser = np.max(subspace.shape)
                conv_output_size = (n_laser - 1) // 2 + 1
                conv_output_size = (conv_output_size - 1) // 2 + 1
                #total_concat_size += 32 * conv_output_size
                total_concat_size += 512

                extractors[key] = nn.Sequential(nn.Conv1d(in_channels=5, out_channels=32, kernel_size=5, stride=2, padding=1, padding_mode="circular"),
                                                nn.ReLU(),
                                                nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, padding_mode="circular"),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                nn.Linear(in_features=4096, out_features=512))             
                
            elif key == "data":
                # Run through a simple MLP
                extractors[key] = nn.Sequential(nn.Flatten())
                total_concat_size += np.prod(subspace.shape)

        self.extractors = nn.ModuleDict(extractors)
        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            x = observations[key]
            x = extractor(x)
            x = x.view(x.shape[0],-1)
            encoded_tensor_list.append(x)

        '''# self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            x = observations[key]
            if key == "laser":
                #pass
                x = x.unsqueeze(1)  # Add a channel dimension
                x = x.view(x.size(0))
            print(x.shape)
            encoded_tensor = extractor(x)
            encoded_tensor_list.append(encoded_tensor)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        print(encoded_tensor_list[0].shape)
        print(encoded_tensor_list[1].shape)'''
        return th.cat(encoded_tensor_list, dim=1)
