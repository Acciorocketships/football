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
from football.models.vanilla_model import VanillaModelConfig
from football.models.deepset_model import DeepSetModelConfig
from football.models.default_model import DefaultModelConfig
from football.algorithms.ddpg import DdpgConfig
from football.util.render_function import render_callback
from football.util.state_predictor import StatePredictorCallback

def update_registries():
    benchmarl.models.model_config_registry.update({
        "vanilla_model": VanillaModelConfig,
        "deepset_model": DeepSetModelConfig,
        "default_model": DefaultModelConfig,
    })
    benchmarl.algorithms.algorithm_config_registry.update({
        "ddpg": DdpgConfig,
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

    VmasTask.render_callback = render_callback

    experiment = Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=cfg.seed,
        config=experiment_config,
        # callbacks=[StatePredictorCallback()],
    )
    return experiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    experiment = get_experiment(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    update_registries()
    hydra_experiment()
