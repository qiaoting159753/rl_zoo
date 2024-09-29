from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from agents.networks.world_models import World_Model
from utils.helpers import normalize_observation_delta, denormalize_observation_delta
import logging
import sys
from agents.networks.world_models.deterministic import (
    Probabilistic_SAS_Reward,
)
from typing import Any, Callable
from agents.networks.world_models.bayesian.bayesian_aleximmer_utils import *

from torch.nn.utils import parameters_to_vector, vector_to_parameters

class KronLaplace:
    """Laplace approximation with Kronecker factored log likelihood Hessian approximation
    and hence posterior precision.
    Mathematically, we have for each parameter group, e.g., torch.nn.Module,
    that \\P\\approx Q \\otimes H\\.
    See `BaseLaplace` for the full interface and see
    `laplace.utils.matrix.Kron` and `laplace.utils.matrix.KronDecomposed` for the structure of
    the Kronecker factors. `Kron` is used to aggregate factors by summing up and
    `KronDecomposed` is used to add the prior, a Hessian factor (e.g. temperature),
    and computing posterior covariances, marginal likelihood, etc.
    Damping can be enabled by setting `damping=True`.
    """

    # key to map to correct subclass of BaseLaplace, (subset of weights, Hessian structure)
    _key = ("all", "kron")

    def __init__(
            self,
            model: nn.Module,
            likelihood: str,
            sigma_noise: float | torch.Tensor = 1.0,
            prior_precision: float | torch.Tensor = 1.0,
            prior_mean: float | torch.Tensor = 0.0,
            temperature: float = 1.0,
            enable_backprop: bool = False,
            dict_key_x: str = "input_ids",
            dict_key_y: str = "labels",
            backend: type[CurvatureInterface] | None = None,
            backend_kwargs: dict[str, Any] | None = None,
            asdl_fisher_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._device = 'cpu'
        self.model: nn.Module = model
        self.likelihood: 'regression'

        # Only do Laplace on params that require grad
        self.params: list[torch.Tensor] = []
        self.is_subset_params: bool = False
        for p in model.parameters():
            if p.requires_grad:
                self.params.append(p)
            else:
                self.is_subset_params = True

        self.n_params: int = sum(p.numel() for p in self.params)
        self.n_layers: int = len(self.params)
        self.prior_precision: float | torch.Tensor = prior_precision
        self.prior_mean: float | torch.Tensor = prior_mean
        self.sigma_noise: float | torch.Tensor = sigma_noise
        self.temperature: float = temperature
        self.enable_backprop: bool = enable_backprop

        # For models with dict-like inputs (e.g. Huggingface LLMs)
        self.dict_key_x = dict_key_x
        self.dict_key_y = dict_key_y

        if backend is None:
            backend = CurvlinopsGGN
        else:
            if self.is_subset_params and (
                    "backpack" in backend.__name__.lower()
                    or "asdfghjkl" in backend.__name__.lower()
            ):
                raise ValueError(
                    "If some grad are switched off, the BackPACK and Asdfghjkl backends"
                    " are not supported."
                )

        self._backend: CurvatureInterface | None = None
        self._backend_cls: type[CurvatureInterface] = backend
        self._backend_kwargs: dict[str, Any] = (
            dict() if backend_kwargs is None else backend_kwargs
        )
        self._asdl_fisher_kwargs: dict[str, Any] = (
            dict() if asdl_fisher_kwargs is None else asdl_fisher_kwargs
        )

        # log likelihood = g(loss)
        self.loss: float = 0.0
        self.n_outputs: int = 0
        self.n_data: int = 0

        # Declare attributes
        self._prior_mean: torch.Tensor
        self._prior_precision: torch.Tensor
        self._sigma_noise: torch.Tensor
        self._posterior_scale: torch.Tensor | None
        if not hasattr(self, "H"):
            self._init_H()
            # posterior mean/mode
            self.mean: float | torch.Tensor = self.prior_mean

    @property
    def log_det_prior_precision(self) -> torch.Tensor:
        """Compute log determinant of the prior precision
        \\(\\log \\det P_0\\)

        Returns
        -------
        log_det : torch.Tensor
        """
        return self.prior_precision_diag.log().sum()

    @property
    def prior_precision(self) -> torch.Tensor:
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, prior_precision: float | torch.Tensor):
        self._posterior_scale = None

        if np.isscalar(prior_precision) and np.isreal(prior_precision):
            self._prior_precision = torch.tensor([prior_precision], device=self._device)
        elif isinstance(prior_precision, torch.Tensor):
            if prior_precision.ndim == 0:
                # make dimensional
                self._prior_precision = prior_precision.reshape(-1).to(self._device)
            elif prior_precision.ndim == 1:
                if len(prior_precision) not in [1, self.n_layers, self.n_params]:
                    raise ValueError(
                        "Length of prior precision does not align with architecture."
                    )
                self._prior_precision = prior_precision.to(self._device)
            else:
                raise ValueError(
                    "Prior precision needs to be at most one-dimensional tensor."
                )
        else:
            raise ValueError(
                "Prior precision either scalar or torch.Tensor up to 1-dim."
            )

    @property
    def prior_precision_diag(self) -> torch.Tensor:
        """Obtain the diagonal prior precision \\(p_0\\) constructed from either
        a scalar, layer-wise, or diagonal prior precision.

        Returns
        -------
        prior_precision_diag : torch.Tensor
        """
        prior_prec: torch.Tensor = (
            self.prior_precision
            if isinstance(self.prior_precision, torch.Tensor)
            else torch.tensor(self.prior_precision)
        )

        if prior_prec.ndim == 0 or len(prior_prec) == 1:  # scalar
            return self.prior_precision * torch.ones(self.n_params, device=self._device)
        elif len(prior_prec) == self.n_params:  # diagonal
            return prior_prec
        elif len(prior_prec) == self.n_layers:  # per layer
            n_params_per_layer = [p.numel() for p in self.params]
            return torch.cat(
                [
                    prior * torch.ones(n_params, device=self._device)
                    for prior, n_params in zip(prior_prec, n_params_per_layer)
                ]
            )
        else:
            raise ValueError(
                "Mismatch of prior and model. Diagonal, scalar, or per-layer prior."
            )

    def _init_H(self) -> None:
        self.H = Kron.init_from_model(
            self.params, self._device
        )

    def _check_H_init(self):
        if getattr(self, "H_facs", None) is None:
            raise AttributeError("Laplace not fitted. Run fit() first.")

    def _curv_closure(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
            N: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backend.kron(X, y, N=N, **self._asdl_fisher_kwargs)

    @staticmethod
    def _rescale_factors(kron: Kron, factor: float) -> Kron:
        for F in kron.kfacs:
            if len(F) == 2:
                F[1] *= factor
        return kron

    def __call__(
            self,
            x: torch.Tensor | MutableMapping[str, torch.Tensor | Any],
            pred_type: PredType | str = PredType.GLM,
            joint: bool = False,
            link_approx: LinkApprox | str = LinkApprox.PROBIT,
            n_samples: int = 100,
            diagonal_output: bool = False,
            generator: torch.Generator | None = None,
            fitting: bool = False,
            **model_kwargs: dict[str, Any],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if generator is not None:
            if (
                    not isinstance(generator, torch.Generator)
                    or generator.device != self._device
            ):
                raise ValueError("Invalid random generator (check type and device).")

        samples = self._nn_predictive_samples(x, n_samples, **model_kwargs)
        return samples.mean(dim=0), samples.var(dim=0)

    def _nn_predictive_samples(
            self,
            X: torch.Tensor,
            n_samples: int = 100,
            generator: torch.Generator | None = None,
            **model_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        fs = list()
        for sample in self.sample(n_samples, generator):
            vector_to_parameters(sample, self.params)
            logits = self.model(
                X.to(self._device) if isinstance(X, torch.Tensor) else X, **model_kwargs
            )
            fs.append(logits.detach() if not self.enable_backprop else logits)
        vector_to_parameters(self.mean, self.params)
        fs = torch.stack(fs)
        return fs

    def sample(self, n_samples: int = 100, generator: torch.Generator | None = None) -> torch.Tensor:
        samples = torch.randn(
            n_samples, self.n_params, device=self._device, generator=generator
        )
        samples = self.posterior_precision.bmm(samples, exponent=-0.5)
        return self.mean.reshape(1, self.n_params) + samples.reshape(
            n_samples, self.n_params
        )


























class Linear_2L_KFRA(nn.Module):
    def __init__(self, input_dim, output_dim, n_hid):
        super(Linear_2L_KFRA, self).__init__()
        self.n_hid = n_hid
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, self.n_hid)
        self.fc2 = nn.Linear(self.n_hid, self.n_hid)
        self.fc3 = nn.Linear(self.n_hid, output_dim)
        # choose your non linearity
        self.act = nn.ReLU(inplace=True)
        self.one = None
        self.a2 = None
        self.h2 = None
        self.a1 = None
        self.h1 = None
        self.a0 = None

    def forward(self, x):
        self.one = x.new(x.shape[0], 1).fill_(1)
        a0 = x.view(-1, self.input_dim)  # view(batch_size, input_dim)
        self.a0 = torch.cat((a0.data, self.one), dim=1)
        # -----------------
        h1 = self.fc1(a0)
        self.h1 = h1.data  # torch.cat((h1, self.one), dim=1)
        # -----------------
        a1 = self.act(h1)
        #         a1.retain_grad()
        self.a1 = torch.cat((a1.data, self.one), dim=1)
        # -----------------
        h2 = self.fc2(a1)
        self.h2 = h2.data  # torch.cat((h2, self.one), dim=1)
        # -----------------
        a2 = self.act(h2)
        #         a2.retain_grad()
        self.a2 = torch.cat((a2.data, self.one), dim=1)
        # -----------------
        h3 = self.fc3(a2)
        return h3


class Bayesian_Laplace(World_Model):
    def __init__(self,
                 observation_size,
                 num_actions,
                 l_r,
                 hidden_size,
                 device):
        self.statistics = None
        self.device = device
        self.observation_size = observation_size
        self.world_model = Linear_2L_KFRA(input_dim=observation_size + num_actions, output_dim=2 * observation_size,
                                          n_hid=128)
        self.world_optimizers = torch.optim.Adam(self.world_model.parameters(), lr=l_r, betas=(0.9, 0.999), eps=1e-08)
        self.reward_model = Probabilistic_SAS_Reward(observation_size=observation_size, num_actions=num_actions,
                                                     hidden_size=hidden_size)
        self.reward_optimizers = torch.optim.Adam(self.reward_model.parameters(), lr=l_r)
        self.reward_model.to(self.device)
        self.world_model.to(self.device)

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.5,
        #                                  weight_decay=(1 / self.prior_sig ** 2))

    def set_statistics(self, statistics: dict) -> None:
        """
        Update all statistics for normalization for all world models and the
        ensemble itself.

        :param (Dictionary) statistics:
        """
        for key, value in statistics.items():
            if isinstance(value, np.ndarray):
                statistics[key] = torch.FloatTensor(statistics[key]).to(self.device)
        self.statistics = statistics
        self.world_model.statistics = statistics

    def train_world(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
    ) -> None:
        target = next_states - states
        delta_targets_normalized = normalize_observation_delta(target, self.statistics)
        s_n_a = torch.cat((states, actions), dim=1)
        pred_s = self.world_model.forward(s_n_a)
        n_mean_delta = pred_s[:, self.observation_size:]
        n_log_delta = pred_s[:, :self.observation_size]
        logvar = torch.tanh(n_log_delta)
        normalized_var = torch.exp(logvar)
        model_loss = F.gaussian_nll_loss(input=n_mean_delta, target=delta_targets_normalized, var=normalized_var).mean()
        self.world_optimizers.zero_grad()
        model_loss.backward()
        self.world_optimizers.step()

    def pred_next_states(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_n_a = torch.cat((observation, actions), dim=1)
        pred_s = self.world_model.forward(s_n_a)
        n_mean_delta = pred_s[:, self.observation_size:]
        n_log_delta = pred_s[:, :self.observation_size]
        logvar = torch.tanh(n_log_delta)
        normalized_var = torch.exp(logvar)
        if torch.any(torch.isnan(n_mean_delta)):
            logging.info("Predicting all Nans")
            sys.exit()
        prediction = denormalize_observation_delta(n_mean_delta, self.statistics)
        prediction += observation
        return prediction, None, n_mean_delta, normalized_var

    def pred_rewards(self, observation: torch.Tensor,
                     action: torch.Tensor, next_observation: torch.Tensor):
        """
        predict reward based on current observation and action and next state
        """
        pred_reward, reward_var = self.reward_model.forward(observation, action, next_observation)
        return pred_reward, None, reward_var

    def train_reward(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            rewards: torch.Tensor,
    ) -> None:
        """
        Train the reward with S, A, SN to eliminate difference between them.
        Args:
            states:
            actions:
            next_states:
            rewards:
        """
        self.reward_optimizers.zero_grad()
        rwd_mean, rwd_var = self.reward_model.forward(states, actions, next_states)
        # reward_loss = F.mse_loss(rwd_mean, sub_rewards)
        reward_loss = F.gaussian_nll_loss(input=rwd_mean, target=rewards, var=rwd_var).mean()
        reward_loss.backward()
        self.reward_optimizers.step()

    def estimate_uncertainty(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[float, float]:

        self.world_model.eval()



        n_mean_delta = pred_s[:, self.observation_size:]
        n_log_delta = pred_s[:, :self.observation_size]
        logvar = torch.tanh(n_log_delta)
        normalized_var = torch.exp(logvar)
        sample1 = torch.distributions.Normal(n_mean_delta, normalized_var).sample([10])
        sample1 = torch.reshape(sample1, (100, self.observation_size))
        prediction = denormalize_observation_delta(sample1, self.statistics)
        prediction += observation
        uncert = torch.sum(torch.var(prediction, dim=0))
        return uncert.item(), 0.0







