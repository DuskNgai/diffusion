import math

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from diffusers.utils.torch_utils import randn_tensor
import torch

from .formulation_table import FunctionType


class ContinuousTimeNoiseScheduler(SchedulerMixin, ConfigMixin):
    """
    Implements general sampler for continuous time diffusion models, whose forward process is defined as:
        `x_t = scale(t) * x_0 + sigma(t) * noise`.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        t_min (`float`):
            Minimum time parameter for the forward diffusion process.
        t_max (`float`):
            Maximum time parameter for the forward diffusion process.
        sigma_data (`float`, defaults to 1.0):
            The (estimated) standard deviation of the training data.
            E.g., the normal distribution `N(mean_data, sigma_data ** 2 * I)` that is closest to the data distribution.
        scale_fn (`FunctionType`):
            The scale function for the noisy data. This was set to scale(t) = 1.
        scale_deriv_fn (`FunctionType`):
            The derivative of the scale function for the noisy data. This was set to scale'(t) = 0.
        sigma_fn (`FunctionType`):
            The noise level for the noisy data. This was set to sigma(t) = t.
        sigma_deriv_fn (`FunctionType`):
            The derivative of the noise level for the noisy data. This was set to sigma'(t) = 1.
        nsr_inv_fn (`FunctionType`):
            The inverse of the noise-to-signal ratio (nsr) function, which is nsr(t) = sigma(t) / scale(t).
            This was set to nsr_inv(nsr) = nsr^{-1}(nsr) = nsr.
        prediction_type (`str`, defaults to `epsilon`):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample) or `velocity` (predicts the velocity of the noisy sample
            moving in the data space during denoising).
        algorithm_type (`str`, defaults to `ode`):
            Algorithm type for the solver; can be `ode` or `sde`.
        timestep_schedule (`str`, defaults to `linear_lognsr`):
            The schedule for the timesteps; can be `linear_lognsr`, `cosine_lognsr`, `power_lognsr`, or `uniform`.
        kwargs:
            Additional keyword arguments for compatibility.
    """

    @register_to_config
    def __init__(
        self,
        t_min: float = 0.002,
        t_max: float = 80.0,
        sigma_data: float = 1.0,
        scale_fn: FunctionType = lambda t: 1.0,
        scale_deriv_fn: FunctionType = lambda t: 0.0,
        sigma_fn: FunctionType = lambda t: t,
        sigma_deriv_fn: FunctionType = lambda t: 1.0,
        nsr_inv_fn: FunctionType = lambda nsr: nsr,
        prediction_type: str = "epsilon",
        algorithm_type: str = "ode",
        timestep_schedule: str = "linear_lognsr",
        **kwargs,
    ) -> None:
        assert scale_fn(0.0) == 1.0, "The scale function should be 1.0 at t = 0."
        assert sigma_fn(0.0) == 0.0, "The sigma function should be 0.0 at t = 0."
        assert prediction_type in ["epsilon", "sample", "velocity"], f"Prediction type {prediction_type} is not supported."
        assert algorithm_type in ["ode", "sde"], f"Algorithm type {algorithm_type} is not supported."

        # Setable values
        self.num_inference_steps = None
        self.timesteps = None
        self._step_index = None
        self._begin_index = None

    @property
    def init_noise_sigma(self) -> float:
        # standard deviation of the initial noise distribution
        return (
            self.config.sigma_fn(self.config.t_max) ** 2 + (self.config.scale_fn(self.config.t_max) * self.config.sigma_data) ** 2
        ) ** 0.5

    @property
    def step_index(self) -> int | None:
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @step_index.setter
    def step_index(self, step_index: int | None = None) -> None:
        """
        Sets the step index for the scheduler.

        Args:
            step_index (`int`, *optional*):
                The index counter for current timestep.
        """
        self._step_index = step_index

    @property
    def begin_index(self) -> int | None:
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    @begin_index.setter
    def begin_index(self, begin_index: int = 0) -> None:
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_timesteps(self, num_inference_steps: int = None, device: str | torch.device | None = None) -> None:
        """
        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
        """
        self.num_inference_steps = num_inference_steps

        ramp = torch.linspace(0, 1, self.num_inference_steps)
        if self.config.timestep_schedule == "linear_lognsr":
            timesteps = self._compute_linear_lognsr_timesteps(ramp)
        elif self.config.timestep_schedule == "cosine_lognsr":
            timesteps = self._compute_cosine_lognsr_timesteps(ramp)
        elif self.config.timestep_schedule == "power_lognsr":
            timesteps = self._compute_power_lognsr_timesteps(ramp)
        elif self.config.timestep_schedule == "uniform":
            timesteps = self._compute_uniform_timesteps(ramp)
        else:
            raise ValueError(f"Time schedule {self.config.timestep_schedule} is not supported.")

        self.timesteps = timesteps.to(dtype=torch.float32, device=device)

    def _compute_linear_lognsr_timesteps(self, ramp: torch.Tensor) -> torch.Tensor:
        """Implementation closely follows k-diffusion."""
        nsr_min = self.config.sigma_fn(self.config.t_min) / self.config.scale_fn(self.config.t_min)
        nsr_max = self.config.sigma_fn(self.config.t_max) / self.config.scale_fn(self.config.t_max)

        log_nsr_min = math.log(nsr_min)
        log_nsr_max = math.log(nsr_max)
        nsr = torch.exp(log_nsr_max + ramp * (log_nsr_min - log_nsr_max))
        return self.config.nsr_inv_fn(nsr)

    def _compute_cosine_lognsr_timesteps(self, ramp: torch.Tensor) -> torch.Tensor:
        nsr_min = self.config.sigma_fn(self.config.t_min) / self.config.scale_fn(self.config.t_min)
        nsr_max = self.config.sigma_fn(self.config.t_max) / self.config.scale_fn(self.config.t_max)

        atan_nsr_min = math.atan(nsr_min)
        atan_nsr_max = math.atan(nsr_max)
        nsr = torch.tan(atan_nsr_max + ramp * (atan_nsr_min - atan_nsr_max))
        return self.config.nsr_inv_fn(nsr)

    def _compute_power_lognsr_timesteps(self, ramp: torch.Tensor) -> torch.Tensor:
        """Construct timesteps the noise schedule of Karras et al. (2022)."""
        nsr_min = self.config.sigma_fn(self.config.t_min) / self.config.scale_fn(self.config.t_min)
        nsr_max = self.config.sigma_fn(self.config.t_max) / self.config.scale_fn(self.config.t_max)

        rho = 7
        root_nsr_min = nsr_min ** (1 / rho)
        root_nsr_max = nsr_max ** (1 / rho)
        nsr = (root_nsr_max + ramp * (root_nsr_min - root_nsr_max)) ** rho
        return self.config.nsr_inv_fn(nsr)

    def _compute_uniform_timesteps(self, ramp: torch.Tensor) -> torch.Tensor:
        ts = self.config.t_max + ramp * (self.config.t_min - self.config.t_max)
        return ts

    def first_order_update(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            noise (`torch.Tensor`):
                Noise tensor for the stochastic sampling.

        Returns:
            `torch.Tensor`:
                A previous instance of a sample created by the diffusion process.
        """
        # TODO: 0.0 or self.config.t_min?
        t, u = self.timesteps[self.step_index + 1] if self.step_index < self.num_inference_steps - 1 else 0.0, self.timesteps[self.step_index]
        scale_t, scale_u = self.config.scale_fn(t), self.config.scale_fn(u)
        sigma_t, sigma_u = self.config.sigma_fn(t), self.config.sigma_fn(u)
        x_u = sample

        if self.config.algorithm_type == "ode":
            if self.config.prediction_type == "epsilon":
                scale_ratio = scale_t / scale_u
                x_t = scale_ratio * x_u + (sigma_t - scale_ratio * sigma_u) * model_output
            elif self.config.prediction_type == "sample":
                sigma_ratio = sigma_t / sigma_u
                x_t = sigma_ratio * x_u + (scale_t - sigma_ratio * scale_u) * model_output
            elif self.config.prediction_type == "velocity":
                x_t = x_u + (t - u) * model_output
            else:
                raise ValueError(f"Prediction type {self.config.prediction_type} is not supported.")
        elif self.config.algorithm_type == "sde":
            scale_ratio = scale_t / scale_u
            if self.config.prediction_type == "epsilon":
                x_t = scale_ratio * x_u + \
                      2 * (sigma_t - scale_ratio * sigma_u) * model_output + \
                      ((sigma_u * scale_ratio) ** 2 - sigma_t ** 2) ** 0.5 * noise
            elif self.config.prediction_type == "sample":
                sigma_ratio = sigma_t / sigma_u
                x_t = (2 * sigma_ratio - scale_ratio) * x_u + \
                      2 * (scale_t - sigma_ratio * scale_u) * model_output + \
                      ((sigma_u * scale_ratio) ** 2 - sigma_t ** 2) ** 0.5 * noise
            elif self.config.prediction_type == "velocity":
                x_t = x_u + 2 * (t - u) * model_output + ((sigma_u * scale_ratio) ** 2 - sigma_t ** 2) ** 0.5 * noise
            else:
                raise ValueError(f"Prediction type {self.config.prediction_type} is not supported.")
        else:
            raise ValueError(f"Algorithm type {self.config.algorithm_type} is not supported.")

        return x_t

    # Copied from diffusers.schedulers.scheduling_edm_dpmsolver_multistep.EDMDPMSolverMultistepScheduler._init_step_index
    def _init_step_index(self, timestep: torch.Tensor) -> None:
        """
        Initialize the step_index counter for the scheduler.
        """
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    # Copied from diffusers.schedulers.scheduling_edm_dpmsolver_multistep.EDMDPMSolverMultistepScheduler.index_for_timestep
    def index_for_timestep(self, timestep: torch.Tensor, schedule_timesteps=None) -> int:
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        index_candidates = (schedule_timesteps == timestep).nonzero()

        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            step_index = index_candidates[0].item()

        return step_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> SchedulerOutput | tuple:
        """
        Predict the sample from the previous timestep following the reverse process.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`torch.Tensor`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler")

        if self.step_index is None:
            self._init_step_index(timestep)

        if self.config.algorithm_type == "sde":
            noise = randn_tensor(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
        else:
            noise = None

        prev_sample = self.first_order_update(model_output, sample, noise)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def add_noise(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adds noise to the samples that meet the forward diffusion process,
        i.e., `x_t = scale(t) * x_0 + sigma(t) * noise`.

        Args:
            sample (`torch.Tensor`):
                The samples to add noise to.
            noise (`torch.Tensor`):
                The noises tensor for the forward diffusion process.
            timestep (`torch.Tensor`):
                The timestepw for the forward diffusion process.

        Returns:
            `torch.Tensor`:
                The noisy samples.
            `torch.Tensor`:
                The target of the backward process.
            `torch.Tensor`:
                The scale of the samples.
            `torch.Tensor`:
                The sigma of the noise.
        """

        while timestep.dim() < sample.dim():
            timestep = timestep.unsqueeze(-1)
        scale = self.config.scale_fn(timestep)
        sigma = self.config.sigma_fn(timestep)

        if self.config.prediction_type == "sample":
            target = sample
        elif self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "velocity":
            scale_deriv = self.config.scale_deriv_fn(timestep)
            sigma_deriv = self.config.sigma_deriv_fn(timestep)
            target = scale_deriv * sample + sigma_deriv * noise

        return scale * sample + sigma * noise, target, scale, sigma
