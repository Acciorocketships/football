from benchmarl.experiment.callback import Callback
from tensordict import TensorDictBase

class ActionLoggerCallback(Callback):

	def on_train_end(self, training_td: TensorDictBase, group: str):
		self.experiment.logger.log({
			"action_mag": self.experiment.policy[0].action_mag.item(),
			"noise_mag": self.experiment.policy[0].noise_mag.item(),
			"noise_sigma": self.experiment.policy[0].sigma.item(),
		})
