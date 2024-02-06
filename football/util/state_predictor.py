from typing import Optional, List
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
import torch
from torch import nn
from benchmarl.experiment.callback import Callback
from tensordict import TensorDictBase
from torch.distributions import Normal


class StatePredictor(TensorDictModule):
    def __init__(
        self,
        ensemble: Optional[nn.ModuleList] = None,
        observation_key=("agents", "observation"),
        action_key=("agents", "action"),
        intrinsic_reward_key=("agents", "intrinsic_reward"),
    ):
        """Each network in the ensemble should output a tensor of (*batch, action_size * state_size).
        The outputs of this module will be of shape (*batch, action_size, ensemble_size, state_size).
        """
        in_keys = [observation_key,action_key]
        out_keys = [intrinsic_reward_key]
        super().__init__(module=ensemble, in_keys=in_keys, out_keys=out_keys)
        self.observation_key = observation_key
        self.action_key = action_key
        self.intrinsic_reward_key = intrinsic_reward_key
        self.ensemble = ensemble
        self.mean_epistemic = None
        self.var_epistemic = None
        self.gamma = 0.999

    def forward(self, td: TensorDict):
        mus, logvars = self.predict_next_state(td)
        td.set(self.intrinsic_reward_key, self.intrinsic_reward(mus, logvars))
        return td

    def predict_next_state(self, td):
        state = td.get(self.observation_key)
        action = td.get(self.action_key)
        state_action = torch.cat([state, action], dim=-1).detach()
        return self._predict_next_state(state_action)

    def _predict_next_state(self, state_action):
        *batch_shape, _ = state_action.shape
        mus, logvars = [], []
        for net in self.ensemble:
            # *batch, action_size, state_size, 2
            out = net(state_action).reshape(*batch_shape, -1, 2)
            mu, logvar = out.chunk(2, dim=-1)
            mus.append(mu.squeeze(-1))
            logvars.append(logvar.squeeze(-1))

        # These are all shape (*batch, ensemble_size, obs_size)
        mus = torch.stack(mus, dim=-2)
        logvars = torch.stack(logvars, dim=-2)
        return mus, logvars

    def aleatoric_uncertainty(self, mus, logvars):
        # logvars input should be (*batch, act_size, ensemble_size, obs_size)
        return logvars.exp().mean(dim=-2).mean(dim=-1)

    def epistemic_uncertainty(self, mus, logvars):
        # mus input should be (*batch, act_size, ensemble_size, obs_size)
        return mus.var(dim=-2).mean(dim=-1)

    def intrinsic_reward(self, mus, logvars):
        epistemic = self.epistemic_uncertainty(mus, logvars).detach()
        if self.var_epistemic is None:
            self.var_epistemic = epistemic.var()
            self.mean_epistemic = epistemic.mean()
        rew = (epistemic - self.mean_epistemic) / torch.sqrt(self.var_epistemic)
        self.mean_epistemic = self.gamma * self.mean_epistemic + (1-self.gamma) * epistemic.mean()
        self.var_epistemic = self.gamma * self.var_epistemic + (1-self.gamma) * epistemic.var()
        rew = torch.clamp(rew, -10, 10)
        return rew.unsqueeze(-1)


class StatePredictorCallback(Callback):

    def __init__(self):
        super().__init__()
        self.opt_dict = {}

    def on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        next_observation_key = ("next", group, "observation")
        model = self.experiment.algorithm.state_predictor[group]
        if group not in self.opt_dict:
            self.opt_dict[group] = torch.optim.Adam(model.parameters(), lr=self.experiment.config.lr)
        opt = self.opt_dict[group]
        mus, logvars = model.predict_next_state(batch)
        dists = [
            Normal(mus[..., i, :], torch.exp(logvars[..., i, :]))
            for i in range(mus.shape[-2])
        ]
        loss_full = [
            -dists[i].log_prob(batch.get(next_observation_key))
            for i in range(mus.shape[-2])
        ]
        loss = (sum(loss_full) / len(loss_full)).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        self.experiment.logger.log({
            "state_predictor_loss": loss.item(),
            "intrinsic_reward": batch.get(("next", group, "intrinsic_reward")).mean().item(),
        })
