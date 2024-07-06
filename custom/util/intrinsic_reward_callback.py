from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from benchmarl.experiment.callback import Callback
from custom.util.empowerment_curiosity import EmpowermentCuriosity

class IntrinsicRewards(Callback):
    def __init__(self, experiment, group=None, empowerment_coeff=0, novelty_coeff=0, actor_coeff=0):
        super().__init__()
        self.empowerment_key = "empowerment_intrinsic_reward"
        self.novelty_key = "novelty_intrinsic_reward"
        self.actor_key = "actor_intrinsic_reward"
        self.coeffs = {self.empowerment_key: empowerment_coeff, self.novelty_key: novelty_coeff, self.actor_key: actor_coeff}
        self.group = group if (group is not None) else list(experiment.group_map.keys())[0]
        self.obs_key = (self.group, "observation")
        self.next_obs_key = ("next", self.group, "observation")
        self.action_key = (self.group, "action")
        self.obs_spec = experiment.observation_spec[self.group]['observation']
        self.act_spec = experiment.action_spec[self.group]['action']
        self.device = experiment.config.train_device
        if self.coeffs[self.empowerment_key] != 0:
            mod = EmpowermentCuriosity(obs_spec=self.obs_spec, act_spec=self.act_spec, device=self.device)
            self.empowerment_curiosity = TensorDictModule(
                mod,
                in_keys=[self.next_obs_key],
                out_keys=[self.get_reward_key(self.group, self.empowerment_key)]
            )

    def get_reward_key(self, group, key):
        return ("next", group, key)

    def before_train_step(self, tensordict: TensorDictBase, group):
        if group != self.group:
            return
        if self.coeffs[self.empowerment_key] != 0:
            self.empowerment_curiosity(tensordict)
        # Add intrinsic_rewards to reward
        reward = tensordict.get(self.get_reward_key(self.group, "reward"))
        for intrinsic_reward_key in [self.empowerment_key, self.novelty_key, self.actor_key]:
            full_key = self.get_reward_key(self.group, intrinsic_reward_key)
            if full_key in tensordict:
                reward += self.coeffs[intrinsic_reward_key] * tensordict.get(full_key).view(reward.shape)
        tensordict.set(self.get_reward_key(self.group, "reward"), reward)

    def on_train_step(self, training_td: TensorDictBase, group: str) -> TensorDictBase:
        if group != self.group:
            return training_td
        if self.coeffs[self.empowerment_key] != 0:
            obs = training_td.get(self.obs_key)
            action = training_td.get(self.action_key)
            next_obs = training_td.get(self.next_obs_key)
            self.empowerment_curiosity.train_model(obs, action, next_obs)