#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Type

import torch

from benchmarl.models.common import Model, ModelConfig
from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import MLP, MultiAgentMLP


class VanillaModel(Model):
    def __init__(
        self,
        **kwargs,
    ):
        # Models in BenchMARL are instantiated per agent group.
        # This means that each model will process the inputs for a whole group of agents
        # There are some core attributes that models are created with,
        # which we are now going to illustrate

        # Remember the kwargs to the super() class
        super().__init__(**kwargs)

        # You can create your custom attributes
        self.activation_class = nn.Mish

        # And access some of the ones already available to your module
        _ = self.input_spec  # Like its input_spec
        _ = self.output_spec  # or output_spec
        _ = self.action_spec  # the action spec of the env
        _ = self.agent_group  # the name of the agent group the model is for
        _ = self.n_agents  # or the number of agents this module is for

        # The following are some of the most important attributes of the model.
        # They decide how the model should be used.
        # Since models can be used for actors and critics, the model will behave differently
        # depending on how these attributes are set.
        # BenchMARL will take care of setting these attributes for you, but your role when making
        # a custom model is making sure that all cases are handled properly

        # This tells the model if the input will have a multi-agent dimension or not.
        # For example, the input of policies will always have this set to true,
        # but critics that use a global state have this set to false as the state
        # is shared by all agents
        _ = self.input_has_agent_dim

        # This tells the model if it should have only one set of parameters
        # or a different set of parameters for each agent.
        # This is independent of the other options as it is possible to have different parameters
        # for centralized critics with global input
        _ = self.share_params

        # This tells the model if it has full observability
        # This will always be true when self.input_has_agent_dim==False
        # but in cases where the input has the agent dimension, this parameter is
        # used to distinguish between a decentralised model (where each agent's data
        # is processed separately) and a centralized model, where the model pools all data together
        _ = self.centralised

        # This is a dynamically computed attribute that indicates if the output will have the agent dimension.
        # This will be false when share_params==True and centralised==True, and true in all other cases.
        # When output_has_agent_dim is true, your model's output should contain the multiagent dimension,
        # and the dimension should be absent otherwise
        _ = self.output_has_agent_dim

        self.input_features = self.input_leaf_spec.shape[-1]
        self.output_features = self.output_leaf_spec.shape[-1]

        self.sae_max_n = 5
        self.sae_dim = 4
        self.sae_hidden_dim = 20
        self.sae_n_agents = 3
        self.sae_n_obs = 5

        # Instantiate a model for this scenario
        # This code will be executed for a policy or for a decentralized critic for example
        self.mlp = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=self.centralised,
            share_params=self.share_params,
            device=self.device,
            activation_class=self.activation_class,
        )

    def _perform_checks(self):
        super()._perform_checks()

        # Run some checks
        if self.input_has_agent_dim and self.input_leaf_spec.shape[-2] != self.n_agents:
            raise ValueError(
                "If the MLP input has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the MLP output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Gather in_key
        input = tensordict.get(self.in_key)

        # Input has multi-agent input dimension
        res = self.mlp.forward(input)
        if not self.output_has_agent_dim:
            # If we are here the module is centralised and parameter shared.
            # Thus the multi-agent dimension has been expanded,
            # We remove it without loss of data
            res = res[..., 0, :]

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class VanillaModelConfig(ModelConfig):
    # The config parameters for this class, these will be loaded from yaml

    @staticmethod
    def associated_class():
        # The associated algorithm class
        return VanillaModel
