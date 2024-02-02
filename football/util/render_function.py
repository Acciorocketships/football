import torch
from tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase

visualise = (not torch.cuda.is_available())

def render_callback(experiment, env: EnvBase, data: TensorDictBase):
    # task_name = experiment.task_name
    group = list(experiment.group_map.keys())[0]
    obs_key = (group, "observation")
    action_key = (group, "action")
    env_index = 0
    agent_index = 0
    def critic(td):
        with experiment.losses[group].value_network_params.to_module(experiment.losses[group].value_network):
            return experiment.losses[group].value_network(td)
    # critic = experiment.losses['agents'].value_network
    def actor(td):
        with experiment.losses[group].actor_network_params.to_module(experiment.losses[group].actor_network):
            return experiment.losses[group].actor_network(td)
    # actor = experiment.losses['agents'].actor_network
    def f(pos):
        obs = torch.stack([
            env.scenario.observation_from_pos(torch.tensor(pos, device=data.device), env_index=env_index, agent_index=agent_idx).float()
            for agent_idx in range(data.get(action_key).shape[1])], dim=1
        )
        actor_input = TensorDict({obs_key: obs}, batch_size=pos.shape[0], device=data.device)
        critic_input = actor(actor_input)
        values_dict = critic(critic_input)
        values = values_dict.get((group, "state_action_value"))[:,agent_index]
        return values


    return env.render(
        mode="rgb_array",
        visualize_when_rgb=visualise,
        plot_position_function=f,
        plot_position_function_range=(1.5, 0.75),
        plot_position_function_cmap_alpha=0.5,
        env_index=env_index,
        plot_position_function_precision=0.05,
        # plot_position_function_cmap_range=[0.0, 1.0],
    )
