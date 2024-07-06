import torch
from torch import nn
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torchrl.modules import MultiAgentMLP
from torch import optim

class EmpowermentCuriosity(nn.Module):

	def __init__(self, obs_spec, act_spec, device, n_samples=16):
		super().__init__()
		self.input_features = obs_spec.shape[-1] + act_spec.shape[-1]
		self.output_features = 2 * obs_spec.shape[-1]
		self.n_agents = obs_spec.shape[-2]
		self.Ptrans = MultiAgentMLP(
			n_agent_inputs=self.input_features,
			n_agent_outputs=self.output_features,
			n_agents=self.n_agents,
			centralised=False,
			share_params=True,
			device=device,
			activation_class=nn.Mish,
		)
		self.batchnorm = nn.BatchNorm1d(1, eps=1e-7)
		self.optim = optim.Adam(self.Ptrans.parameters(), lr=1e-4)
		if act_spec.shape[-1] == 2:
			t = torch.arange(n_samples) * (2 * torch.pi) / n_samples
			self.action_samples = torch.stack([torch.cos(t), torch.sin(t)], dim=-1)
			self.action_samples[1::2] *= 0.5
		else:
			raise ValueError("EmpowermentCuriosity action samples not implemented for action dim!=2")

	def mutual_information(self, dist):
		# The mutual information I(x,a) between state x and action a
		# H(x|a) = ∑ P(a=a_i) * H(x|a=a_i)
		Hxa = -dist.log_prob(dist.loc).mean(dim=-1)
		# Hxa = dist.entropy().mean(dim=-1)
		# To approximate H(x) (because it is a mixture of gaussians), we use the 0th order approximation
		# from https://www.researchgate.net/publication/224338003_On_Entropy_Approximation_for_Gaussian_Mixture_Random_Vectors
		weights = Categorical(torch.ones(dist.loc.shape[:-1]))
		mixture_dist = MixtureSameFamily(weights, dist)
		batch_shape = (1,) + tuple(mixture_dist.batch_shape)
		mixture_dist = mixture_dist.expand(batch_shape)
		mus = dist.loc.permute(2,0,1,3)
		Hx = -mixture_dist.log_prob(mus).mean(dim=0)
		# I(x,a) = H(x) - H(x|a)
		Ixa = Hx - Hxa
		return Ixa

	def forward(self, obs):
		obs_expanded = obs.unsqueeze(1).repeat(1,self.action_samples.shape[0],1,1)
		action_expanded = self.action_samples.unsqueeze(0).unsqueeze(2).repeat(obs.shape[0], 1, obs.shape[1], 1)
		obs_action = torch.cat([obs_expanded, action_expanded], dim=-1)
		logits = self.Ptrans(obs_action).view(-1, self.action_samples.shape[0], self.n_agents, self.output_features//2, 2).permute(0,2,1,3,4)
		next_obs_dist = LowRankMultivariateNormal(
			loc=logits[...,0],
			cov_diag=torch.exp(logits[...,1]),
			cov_factor=torch.zeros(logits.shape[:-1]).unsqueeze(-1)
		)
		mi = self.mutual_information(next_obs_dist)
		mi = self.batchnorm(mi.view(-1,1)).view(mi.shape)
		return mi

	def train_model(self, obs, action, next_obs):
		self.optim.zero_grad()
		obs_action = torch.cat([obs, action], dim=-1)
		logits = self.Ptrans(obs_action).view(-1, self.n_agents, self.output_features//2, 2)
		next_obs_dist = LowRankMultivariateNormal(
			loc=logits[..., 0],
			cov_diag=torch.exp(logits[..., 1]),
			cov_factor=torch.zeros(logits.shape[:-1]).unsqueeze(-1)
		)
		loss = -next_obs_dist.log_prob(next_obs).mean()
		loss.backward()
		self.optim.step()
