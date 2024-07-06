#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import benchmarl
from benchmarl.experiment import Experiment
from benchmarl.hydra_config import (
    load_algorithm_config_from_hydra,
    load_experiment_config_from_hydra,
    load_task_config_from_hydra,
    load_model_config_from_hydra,
)

from benchmarl.environments.vmas.common import VmasTask
from custom.models.default_model import DefaultModelConfig
from custom.algorithms.ddpg import DdpgConfig
from custom.algorithms.ppo import PPOConfig
from custom.util.intrinsic_reward_callback import IntrinsicRewards
# from custom.util.render_value_callback import render_callback
from custom.util.log_actions import ActionLoggerCallback

def update_registries():
    benchmarl.models.model_config_registry.update({
        "default_model": DefaultModelConfig,
    })
    benchmarl.algorithms.algorithm_config_registry.update({
        "ddpg": DdpgConfig,
        "ppo": PPOConfig,
    })
    benchmarl._load_hydra_schemas()

def get_experiment(cfg: DictConfig) -> Experiment:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    update_registries()

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    algorithm_config = load_algorithm_config_from_hydra(cfg.algorithm)
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)
    task_config = load_task_config_from_hydra(cfg.task, task_name)
    critic_model_config = load_model_config_from_hydra(cfg.critic_model)
    model_config = load_model_config_from_hydra(cfg.model)

    # VmasTask.render_callback = render_callback
    experiment = Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=cfg.seed,
        config=experiment_config,
        callbacks=[]
    )

    intrinsic_rewards = IntrinsicRewards(experiment, empowerment_coeff=1.)
    experiment.callbacks += [intrinsic_rewards]

    return experiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    experiment = get_experiment(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    update_registries()
    hydra_experiment()
