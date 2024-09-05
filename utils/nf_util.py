import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Move model on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DiagGaussian(nn.Module):
    def __init__(self, shape, trainable=True):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *self.shape))
            self.log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        else:
            self.register_buffer("loc", torch.zeros(1, *self.shape))
            self.register_buffer("log_scale", torch.zeros(1, *self.shape))
        self.temperature = None  # Temperature parameter for annealed sampling

    def forward(self, num_samples=1):
        eps = torch.randn((num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device)
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        z = self.loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1)))
        return z, log_p

    def log_prob(self, z):
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p

    def sample(self, num_samples=1, **kwargs):
        z, _ = self.forward(num_samples, **kwargs)
        return z


class ConditionalDiagGaussian():
    def __init__(self, shape):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)

    def forward(self, mean, log_scale, num_samples=1):
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=mean.dtype, device=mean.device
        )
        z = mean + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p

    def log_prob(self, mean, log_scale, num_samples):
        z, _ = self.forward(mean, log_scale, num_samples)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - mean) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p


class Permute(nn.Module):
    """
    Permutation features along the channel dimension
    """
    def __init__(self, num_channels, mode="shuffle"):
        """Constructor
        Args:
          num_channel: Number of channels
          mode: Mode of permuting features, can be shuffle for random permutation or swap for interchanging upper and lower part
        """
        super().__init__()
        self.mode = mode
        self.num_channels = num_channels
        if self.mode == "shuffle":
            perm = torch.randperm(self.num_channels)
            inv_perm = torch.empty_like(perm).scatter_(
                dim=0, index=perm, src=torch.arange(self.num_channels)
            )
            self.register_buffer("perm", perm)
            self.register_buffer("inv_perm", inv_perm)

    def forward(self, z, context=None):
        if self.mode == "shuffle":
            z = z[:, self.perm, ...]
        elif self.mode == "swap":
            z1 = z[:, : self.num_channels // 2, ...]
            z2 = z[:, self.num_channels // 2 :, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError("The mode " + self.mode + " is not implemented.")
        log_det = torch.zeros(len(z), device=z.device)
        return z, log_det

    def inverse(self, z, context=None):
        if self.mode == "shuffle":
            z = z[:, self.inv_perm, ...]
        elif self.mode == "swap":
            z1 = z[:, : (self.num_channels + 1) // 2, ...]
            z2 = z[:, (self.num_channels + 1) // 2 :, ...]
            z = torch.cat([z2, z1], dim=1)
        else:
            raise NotImplementedError("The mode " + self.mode + " is not implemented.")
        log_det = torch.zeros(len(z), device=z.device)
        return z, log_det


class MaskedLinear(nn.Linear):
    """A linear module with a masked weight matrix."""
    def __init__(
        self,
        in_degrees,
        out_features,
        autoregressive_features,
        random_mask,
        is_output,
        bias=True,
        out_degrees_=None,
    ):
        super().__init__(
            in_features=len(in_degrees), out_features=out_features, bias=bias
        )
        mask, degrees = self._get_mask_and_degrees(
            in_degrees=in_degrees,
            out_features=out_features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=is_output,
            out_degrees_=out_degrees_,
        )
        self.register_buffer("mask", mask)
        self.register_buffer("degrees", degrees)

    @classmethod
    def _get_mask_and_degrees(
        cls,
        in_degrees,
        out_features,
        autoregressive_features,
        random_mask,
        is_output,
        out_degrees_=None,
    ):
        if is_output:
            if out_degrees_ is None:
                out_degrees_ = _get_input_degrees(autoregressive_features)
            out_degrees = tile(out_degrees_, out_features // autoregressive_features)
            mask = (out_degrees[..., None] > in_degrees).float()
        else:
            if random_mask:
                min_in_degree = torch.min(in_degrees).item()
                min_in_degree = min(min_in_degree, autoregressive_features - 1)
                out_degrees = torch.randint(
                    low=min_in_degree,
                    high=autoregressive_features,
                    size=[out_features],
                    dtype=torch.long,
                )
            else:
                max_ = max(1, autoregressive_features - 1)
                min_ = min(1, autoregressive_features - 1)
                out_degrees = torch.arange(out_features) % max_ + min_
            mask = (out_degrees[..., None] >= in_degrees).float()
        return mask, out_degrees

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MaskedFeedforwardBlock(nn.Module):
    def __init__(
        self,
        in_degrees,
        autoregressive_features,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        features = len(in_degrees)
        # Batch norm.
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features, eps=1e-3)
        else:
            self.batch_norm = None

        # Masked linear.
        self.linear = MaskedLinear(
            in_degrees=in_degrees,
            out_features=features,
            autoregressive_features=autoregressive_features,
            random_mask=random_mask,
            is_output=False,
        )

        self.degrees = self.linear.degrees
        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs):
        outputs = inputs
        outputs = self.linear(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs


class MADE(nn.Module):
    """Implementation of MADE.
    It can use either feedforward blocks or residual blocks (default is residual).
    Optionally, it can use batch norm or dropout within blocks (default is no).
    """
    def __init__(
        self,
        features,
        hidden_features,
        num_blocks=2,
        output_multiplier=2,
        use_residual_blocks=True,
        random_mask=False,
        permute_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        if use_residual_blocks and random_mask:
            raise ValueError("Residual blocks can't be used with random masks.")
        super().__init__()

        self.preprocessing = torch.nn.Identity()

        # Initial layer.
        input_degrees_ = _get_input_degrees(features)
        if permute_mask:
            input_degrees_ = input_degrees_[torch.randperm(features)]

        self.initial_layer = MaskedLinear(
            in_degrees=input_degrees_,
            out_features=hidden_features,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=False,
        )

        # Residual blocks.
        blocks = []
        block_constructor = MaskedFeedforwardBlock
        prev_out_degrees = self.initial_layer.degrees
        for _ in range(num_blocks):
            blocks.append(
                block_constructor(
                    in_degrees=prev_out_degrees,
                    autoregressive_features=features,
                    random_mask=random_mask,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
            prev_out_degrees = blocks[-1].degrees
        self.blocks = nn.ModuleList(blocks)

        # Final layer.
        self.final_layer = MaskedLinear(
            in_degrees=prev_out_degrees,
            out_features=features * output_multiplier,
            autoregressive_features=features,
            random_mask=random_mask,
            is_output=True,
            out_degrees_=input_degrees_,
        )

    def forward(self, inputs, context=None):
        outputs = self.preprocessing(inputs)
        outputs = self.initial_layer(outputs)
        if context is not None:
            outputs += self.context_layer(context)
        for block in self.blocks:
            outputs = block(outputs)
        outputs = self.final_layer(outputs)
        return outputs


class MaskedAffineAutoregressive(nn.Module):
    def __init__(
        self,
        features,
        hidden_features,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super(MaskedAffineAutoregressive, self).__init__()
        """Constructor
        Args:
          features: Number of features/input dimensions
          hidden_features: Number of hidden units in the MADE network
          context_features: Number of context/conditional features
          num_blocks: Number of blocks in the MADE network
          use_residual_blocks: Flag whether residual blocks should be used
          random_mask: Flag whether to use random masks
          activation: Activation function to be used in the MADE network
          dropout_probability: Dropout probability in the MADE network
          use_batch_norm: Flag whether batch normalization should be used
        """
        self.autoregressive_net = MADE(
            features=features,
            hidden_features=hidden_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        self.features = features

    def forward(self, inputs, context=None):
        autoregressive_params = self.autoregressive_net(inputs, context)
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        num_inputs = np.prod(inputs.shape[1:])
        # outputs = torch.zeros_like(inputs)
        outputs = inputs
        logabsdet = None
        # When inverse, why it needs to be inverse several times.
        num_inputs = 1
        for _ in range(num_inputs):
            autoregressive_params = self.autoregressive_net(outputs, context)
            outputs, logabsdet = self._elementwise_inverse(inputs, autoregressive_params)
        return outputs, logabsdet

    def _elementwise_forward(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(autoregressive_params)
        scale = torch.sigmoid(unconstrained_scale + 5.0) + 1e-3
        log_scale = torch.log(scale)
        outputs = scale * inputs + shift
        logabsdet = sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(autoregressive_params)
        scale = torch.sigmoid(unconstrained_scale + 5.0) + 1e-3
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        logabsdet = -sum_except_batch(log_scale, num_batch_dims=1)
        return outputs, logabsdet

    def _unconstrained_scale_and_shift(self, autoregressive_params):
        # split_idx = autoregressive_params.size(1) // 2
        # unconstrained_scale = autoregressive_params[..., :split_idx]
        # shift = autoregressive_params[..., split_idx:]
        # return unconstrained_scale, shift
        autoregressive_params = autoregressive_params.view(-1, self.features, 2)
        unconstrained_scale = autoregressive_params[..., 0]
        shift = autoregressive_params[..., 1]
        return unconstrained_scale, shift


#################################      Functions    ##############################################
def _get_input_degrees(in_features):
    """Returns the degrees an input to MADE should have."""
    return torch.arange(1, in_features + 1)


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def tile(x, n):
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'statistics'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def compute_uncertainty(pred_means, pred_vars, solutions=1):
    # Input : [5, 10, 11]
    if solutions == 3:
        print("Don't call me! Using Wrong Solutions !")
        exit(0)

    # Solution 0: Normal sampling
    if solutions == 0:
        # 5 models, each sampled 10 times = 50,
        sample1 = torch.distributions.Normal(pred_means[0], pred_vars[0]).sample([10])
        sample2 = torch.distributions.Normal(pred_means[1], pred_vars[1]).sample([10])
        sample3 = torch.distributions.Normal(pred_means[2], pred_vars[2]).sample([10])
        sample4 = torch.distributions.Normal(pred_means[3], pred_vars[3]).sample([10])
        sample5 = torch.distributions.Normal(pred_means[4], pred_vars[4]).sample([10])
        samples = torch.cat((sample1, sample2, sample3, sample4, sample5), dim=0)
        # Samples = [5 * 10, 10 predictions, 11 state dims]
        stds = torch.std(samples, dim=0)
        # [10 predictions, 11 state dims]
        total_stds = torch.mean(stds, dim=1)
        return total_stds.detach()

    # Solution 1: Epistemic
    if solutions == 1:
        total_stds = torch.std(pred_means, dim=0)
        total_stds = torch.sum(total_stds, dim=1)
        return total_stds.detach()

    # Solution 2: Aleatoric
    if solutions == 2:
        total_stds = torch.sum(pred_vars, dim=0)
        total_stds = torch.sum(total_stds, dim=1)
        return total_stds.detach()


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def get_params(models):
    for m in models:
        for p in m.parameters():
            yield p


def accum_prod(x):
    assert x.dim() == 2
    x_accum = [x[0]]
    for i in range(x.size(0) - 1):
        x_accum.append(x_accum[-1] * x[i])
    x_accum = torch.stack(x_accum, dim=0)
    return x_accum


def init_weights(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)


def normalize_obs(obs, statistics):
    return (obs - statistics["ob_mean"]) / statistics["ob_std"]


def silu(x_input):
    # Applies the Sigmoid Linear Unit (SiLU) function element-wise: SiLU(x) = x * sigmoid(x)
    return x_input * torch.sigmoid(x_input)


def unnormalize_deltas(normalized_deltas, statistics):
    return (normalized_deltas * statistics["delta_std"]) + statistics["delta_mean"]


def normalize_deltas(deltas, statistics):
    return (deltas - statistics["delta_mean"]) / statistics["delta_std"]


def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


def set_requires_grad(module, flag):
    """Sets requires_grad flag of all parameters of a torch.nn.module

    Args:
      module: torch.nn.module
      flag: Flag to set requires_grad to
    """

    for param in module.parameters():
        param.requires_grad = flag
