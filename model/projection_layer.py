import copy
import warnings
import math

import torch
from torch.nn.parameter import Parameter

import model.base
from model.base import BaseModule
from tools import get_filterbank


class ProjectionLayer(BaseModule):
    def __init__(self, config):
        super(ProjectionLayer, self).__init__()
        self.config = config

        self.use_echo_encoder = config.model_config.use_echo_encoder
        # Initializing flag to map latent to RGP
        self.e2s_flag = config.model_config.e2s_flag
        self.s2e_flag = config.model_config.s2e_flag
        assert not self.e2s_flag or self.use_echo_encoder, \
            f"e2s_flag: {self.e2s_flag}, use_echo_encoder: {self.use_echo_encoder}, " \
            f"e2s error (impossible)"
        assert self.s2e_flag or self.use_echo_encoder, \
            f"s2e_flag: {self.s2e_flag}, use_echo_encoder: {self.use_echo_encoder}, " \
            f"s2e error (echo-related empty)"

        self.noise_type = config.model_config.noise_type
        assert self.noise_type in model.base.ALLOWED_NOISE.keys(), \
            f"noise_type: '{self.noise_type}' is not yet supported."
        self.num_noise_segment = config.model_config.num_noise_segment
        self.noise_parameter_partition = \
            torch.tensor(model.base.ALLOWED_NOISE[self.noise_type]) * self.num_noise_segment
        self.num_echo_related_rgp = self.noise_parameter_partition[-1]

        assert config.model_config.synthesis_filterbank_type in model.base.ALLOWED_SYNTHESIS_FILTERBANK_TYPE, \
            f"Invalid synthesis_filterbank_type: '{config.model_config.synthesis_filterbank_type}'"
        self.num_filter = get_filterbank(
            filterbank_type=config.model_config.synthesis_filterbank_type,
            sample_rate=config.data_config.sample_rate,
            signal_length=\
                config.model_config.noise_segment_length * config.model_config.synthesis_filter_length_in_segment,
            synthesis=True
        ).size(-1) if config.model_config.synthesis_filterbank_type != "none" else 1
        self.synthesis_filter_length_in_segment = config.model_config.synthesis_filter_length_in_segment
        self.synthesis_filter_stride_in_segment = config.model_config.synthesis_filter_stride_in_segment
        if not self.synthesis_filter_length_in_segment % self.synthesis_filter_stride_in_segment == 0:
            warnings.warn(
                f"synthesis_filter_stride should be the divisor of synthesis_filter_length. "
                f"nonuniform overlapping occurs."
            )
        self.num_synthesis_window = \
            math.ceil(self.num_noise_segment / self.synthesis_filter_length_in_segment) \
            * self.synthesis_filter_length_in_segment \
            + self.synthesis_filter_length_in_segment \
            - self.synthesis_filter_stride_in_segment
        self.num_spec_related_rgp = self.num_filter * self.num_synthesis_window

        self.num_total_rgp = self.num_spec_related_rgp + self.num_echo_related_rgp
        assert self.num_total_rgp == self.config.model_config.num_rgp, \
            f"{self.num_total_rgp} != total # of RGPs: {self.config.model_config.num_rgp}."

        # Initializing spectral encoder
        self.config_hadamard = copy.deepcopy(config)
        self.config_hadamard.model_config.num_rgp = \
            self.num_spec_related_rgp + self.num_echo_related_rgp * config.model_config.s2e_flag
        self.linear_stage_1 = torch.nn.Linear(
            config.model_config.encoder_tgru_channel * (1 + config.model_config.encoder_tgru_bidirectional),
            self.config_hadamard.model_config.num_rgp
        )
        self.linear_stage_2 = HadamardLinear(self.config_hadamard)

        # Initializing echo encoder
        if self.use_echo_encoder:
            self.config_echo_hadamard = copy.deepcopy(config)
            self.config_echo_hadamard.model_config.encoder_after_conv_time = \
                config.model_config.echo_encoder_after_conv_time
            self.config_echo_hadamard.model_config.num_rgp = \
                self.num_echo_related_rgp + self.num_spec_related_rgp * config.model_config.e2s_flag
            self.from_echo_encoder_linear_stage_1 = torch.nn.Linear(
                self.config.model_config.echo_encoder_tgru_channel *
                (1 + self.config.model_config.echo_encoder_tgru_bidirectional),
                self.config_echo_hadamard.model_config.num_rgp
            )
            self.from_echo_encoder_linear_stage_2 = HadamardLinear(self.config_echo_hadamard)

        self.softplus = torch.nn.Softplus()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)

    def forward(self, spectral_latent, echo_latent=None):
        rgp = {}

        spectral_latent = self.linear_stage_1(spectral_latent)
        spectral_latent = self.linear_stage_2(spectral_latent.transpose(-1, -2))

        if self.use_echo_encoder:
            echo_latent = self.from_echo_encoder_linear_stage_1(echo_latent)
            echo_latent = self.from_echo_encoder_linear_stage_2(echo_latent.transpose(-1, -2))
            assert spectral_latent.size(0) == echo_latent.size(0)

        # Spectral-related RGP
        rgp['filterbank_weight'] = (
            spectral_latent[..., :self.num_spec_related_rgp]
            + (echo_latent[..., self.num_echo_related_rgp:self.num_total_rgp] if self.e2s_flag else 0.)
        ).view(spectral_latent.size(0), self.num_synthesis_window, self.num_filter)
        rgp['filterbank_weight'] = self.softplus(rgp['filterbank_weight']) / math.log(2)

        # Echo-related RGP
        if self.noise_type == "gaussian":
            rgp['gain'] = (
                (
                    spectral_latent[
                        ...,
                        self.num_spec_related_rgp + self.noise_parameter_partition[0]:
                        self.num_spec_related_rgp + self.noise_parameter_partition[1]
                    ] if self.s2e_flag else 0.
                ) + (
                    echo_latent[..., self.num_echo_related_rgp:self.num_total_rgp] if self.e2s_flag else 0.
                )
            ).view(spectral_latent.size(0), self.num_noise_segment, -1)
        elif self.noise_type == "conv-velvet-gaussian":
            rgp['velvet_log_gain'] = (
                (
                    spectral_latent[
                        ...,
                        self.num_spec_related_rgp + self.noise_parameter_partition[0]:
                        self.num_spec_related_rgp + self.noise_parameter_partition[1]
                    ] if self.s2e_flag else 0.
                ) + (
                    echo_latent[
                        ...,
                        self.noise_parameter_partition[0]:self.noise_parameter_partition[1]
                    ] if self.use_echo_encoder else 0.
                )
            ).view(spectral_latent.size(0), self.num_noise_segment, -1)
            rgp['velvet_log_gain'] = self.leaky_relu(rgp['velvet_log_gain']) - 1.
            rgp['gaussian_log_gain'] = (
                (
                    spectral_latent[
                        ...,
                        self.num_spec_related_rgp + self.noise_parameter_partition[1]:
                        self.num_spec_related_rgp + self.noise_parameter_partition[2]
                    ] if self.s2e_flag else 0.
                ) + (
                    echo_latent[
                        ...,
                        self.noise_parameter_partition[1]:self.noise_parameter_partition[2]
                    ] if self.use_echo_encoder else 0.
                )
            ).view(spectral_latent.size(0), self.num_noise_segment, -1)
            rgp['gaussian_log_gain'] = self.leaky_relu(rgp['gaussian_log_gain']) - 2.
        else:
            raise NotImplementedError

        return rgp


class HadamardLinear(BaseModule):
    """
    (Description is under construction)
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, config, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HadamardLinear, self).__init__()
        self.config = config
        self.in_features = self.config.model_config.encoder_after_conv_time
        self.out_features = self.config.model_config.num_rgp

        self.weight = Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
        self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, P, T]
        assert len(x.size()) == 3, f"input must have 3 dimensions, got {len(x.size())}."
        assert x.size(-1) == self.weight.size(-1) and x.size(-2) == self.weight.size(-2), \
            f"input and weight shapes cannot be Hadamard linear computed ({x.size()} and {self.weight.size()})"

        # [B, P]
        hadamard_output = \
            (x * self.weight.unsqueeze(0).expand(x.size(0), -1, -1)).sum(-1) \
            + self.bias.unsqueeze(0).expand(x.size(0), -1)
        return hadamard_output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
