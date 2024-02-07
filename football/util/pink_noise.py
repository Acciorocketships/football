import torch
from typing import Optional
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictModuleWrapper
from tensordict.utils import NestedKey
from torchrl.data.tensor_specs import CompositeSpec, TensorSpec
from torchrl.envs.utils import exploration_type, ExplorationType
from torchrl.modules.tensordict_module.common import _forward_hook_safe_action
from torch.fft import irfft, rfftfreq


class PinkNoiseWrapper(TensorDictModuleWrapper):

    def __init__(
        self,
        policy: TensorDictModule,
        batch_size: int,
        seq_len: int,
        *,
        sigma_init: float = 1.0,
        sigma_end: float = 0.1,
        annealing_num_steps: int = 1000,
        random_num_steps: int = 0,
        action_key: Optional[NestedKey] = "action",
        spec: Optional[TensorSpec] = None,
        safe: Optional[bool] = True,
    ):
        super().__init__(policy)
        if sigma_end > sigma_init:
            raise RuntimeError("sigma should decrease over time or be constant")
        self.register_buffer("sigma_init", torch.tensor([sigma_init]))
        self.register_buffer("sigma_end", torch.tensor([sigma_end]))
        self.annealing_num_steps = annealing_num_steps
        self.random_num_steps = random_num_steps
        self.register_buffer("sigma", torch.tensor([sigma_init], dtype=torch.float32))
        self.action_key = action_key
        self.out_keys = list(self.td_module.out_keys)
        if action_key not in self.out_keys:
            raise RuntimeError(
                f"The action key {action_key} was not found in the td_module out_keys {self.td_module.out_keys}."
            )
        if spec is not None:
            if not isinstance(spec, CompositeSpec) and len(self.out_keys) >= 1:
                spec = CompositeSpec({action_key: spec}, shape=spec.shape[:-1])
            self._spec = spec
        elif hasattr(self.td_module, "_spec"):
            self._spec = self.td_module._spec.clone()
            if action_key not in self._spec.keys(True, True):
                self._spec[action_key] = None
        elif hasattr(self.td_module, "spec"):
            self._spec = self.td_module.spec.clone()
            if action_key not in self._spec.keys(True, True):
                self._spec[action_key] = None
        else:
            self._spec = CompositeSpec({key: None for key in policy.out_keys})

        self.safe = safe
        if self.safe:
            self.register_forward_hook(_forward_hook_safe_action)
        self.pink_noise = ColoredNoiseProcess(size=(batch_size,)+tuple(self._spec[self.action_key].shape), seq_len=seq_len)
        self.pink_noise_eval = None
        self.noise_mag = torch.zeros(1)
        self.action_mag = torch.zeros(1)

    @property
    def spec(self):
        return self._spec

    def step(self, frames: int = 1) -> None:
        self.sigma.data[0] = max(
            self.sigma_end.item(),
            (self.sigma - frames * (self.sigma_init - self.sigma_end) / self.annealing_num_steps).item(),
        )
        self.random_num_steps -= 1

    def _add_noise(self, action: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma.item()
        if action.shape[0] != self.pink_noise.size[0]:
            if self.pink_noise_eval is None:
                self.pink_noise_eval = ColoredNoiseProcess(size=(action.shape[0],) + tuple(self._spec[self.action_key].shape), seq_len=self.pink_noise.time_steps)
            noise = self.pink_noise_eval.sample(1)[0].to(action.device)
        else:
            noise = self.pink_noise.sample(1)[0].to(action.device)
        self.noise_mag = (noise * sigma).norm(dim=-1).mean()
        self.action_mag = action.norm(dim=-1).mean()
        if self.random_num_steps > 0:
            action = noise * sigma
        else:
            action = action + noise * sigma
        spec = self.spec
        spec = spec[self.action_key]
        if spec is not None:
            action = spec.project(action)
        elif self.safe:
            raise RuntimeError(
                "the action spec must be provided to AdditiveGaussianWrapper unless "
                "the `safe` keyword argument is turned off at initialization."
            )
        return action

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = self.td_module.forward(tensordict)
        if exploration_type() is ExplorationType.RANDOM or exploration_type() is None:
            out = tensordict.get(self.action_key)
            out = self._add_noise(out)
            tensordict.set(self.action_key, out)
        return tensordict


def powerlaw_psd_gaussian(exponent, size, fmin=0):
    """Gaussian (1/f)**beta noise.

    exponent : float
        The power-spectrum of the generated noise is proportional to

        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2

        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.

    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.

    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper.

        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.

    """

    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)

    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1./samples)    # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    # Build scaling factors for all frequencies
    s_scale = f
    ix = torch.sum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].clone()
    w[-1] *= (1 + (samples % 2)) / 2.    # correct f = +-0.5
    sigma = 2 * torch.sqrt(torch.sum(w**2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(None,) * dims_to_add + (Ellipsis,)]

    # Generate scaled random power + phase
    sr = torch.randn(size) * s_scale
    si = torch.randn(size) * s_scale

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si[..., -1] = 0
        sr[..., -1] *= torch.sqrt(torch.tensor(2.))   # Fix magnitude

    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0
    sr[..., 0] *= torch.sqrt(torch.tensor(2.))    # Fix magnitude

    # Combine power + corrected phase to Fourier components
    s = sr + 1J * si

    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, dim=-1) / sigma

    return y

class ColoredNoiseProcess():

    def __init__(self, size, seq_len, scale=1, beta=1):

        self.beta = beta
        self.minimum_frequency = 0
        self.scale = scale

        # The last component of size is the time index
        try:
            self.size = list(size)
        except TypeError:
            self.size = [size]
        self.size += [seq_len]
        self.time_steps = seq_len

        # Fill buffer and reset index
        self.reset()

    def reset(self):
        # Reset the buffer with a new time series.
        self.buffer = powerlaw_psd_gaussian(
                exponent=self.beta, size=self.size, fmin=self.minimum_frequency)
        self.idx = 0

    def sample(self, T=1):
        n = 0
        ret = []
        while n < T:
            if self.idx >= self.time_steps:
                self.reset()
            m = min(T - n, self.time_steps - self.idx)
            ret.append(self.buffer[..., self.idx:(self.idx + m)])
            n += m
            self.idx += m

        ret = self.scale * torch.cat(ret, dim=-1)
        return ret.permute(-1, *torch.arange(ret.dim()-1))


if __name__ == "__main__":
    length = 100
    pink_noise = ColoredNoiseProcess(size=2, seq_len=length)
    x = pink_noise.sample(length)
    X = torch.cumsum(x, dim=0)
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    plt.plot(X[:,0], X[:,1])
    plt.show()