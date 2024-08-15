import torch

import model.base
from model.base import BaseModule
from model.nn import Echo2ReverbNN
from tools import differentiable_ned, normalized_echo_density, hpf_before_get_echo_density


class Echo2Reverb(BaseModule):
    def __init__(self, config):
        super(Echo2Reverb, self).__init__()
        self.config = config
        self.nn = Echo2ReverbNN(config)
        offset = config.model_config.echo_density_offset
        rir_length_for_echo_density = \
            int(config.data_config.rir_length_in_ms * config.data_config.sample_rate / 1000) - offset
        self.n_win_compute = \
            lambda x: (x - config.model_config.echo_density_window_length) // config.model_config.echo_density_stride
        n_win_for_echo_density = max(self.n_win_compute(rir_length_for_echo_density), 0) + 1
        self.kwargs = dict(
            segment_length=config.model_config.echo_density_window_length,
            stride=config.model_config.echo_density_stride,
            n_win=n_win_for_echo_density,
            offset=offset,
            window_type=config.model_config.echo_density_window
        )
        self.loss_kwargs = self.kwargs.copy()
        self.loss_kwargs['slope'] = config.model_config.normalized_echo_density_sigmoid_slope
        self.init_flags(False)

    def init_flags(self, export_misc=False):
        self.nn.reverberation_generator.register_buffer("export_misc", torch.tensor(export_misc))

    def compute_loss(self, rir_ground_truth, input_data, mel_spec_func_list):
        losses = {}
        rir_ground_truth = rir_ground_truth.squeeze(1)
        outputs = self.nn(
            x=input_data,
            early_rir=input_data.squeeze(1),
            h0=input_data[..., :self.config.model_config.rir_direct_length]
        )
        rir_generated = outputs[0]
        losses['multi_resolution_spectral_loss'] = self.multi_resolution_spectral_loss(
            rir_ground_truth, rir_generated, mel_spec_func_list
        ) if self.config.model_config.use_multi_resolution_spectral_loss else torch.tensor(0.)
        losses['normalized_echo_density_loss'] = self.normalized_echo_density_loss(
            rir_ground_truth, rir_generated, **self.loss_kwargs
        ) if self.config.model_config.use_normalized_echo_density_loss else torch.tensor(0.)
        losses['diffuse_insensitive_ned_error'] = self.diffuse_insensitive_ned_error(
            rir_ground_truth, rir_generated, **self.kwargs
        )
        losses['total_loss'] = \
            losses['multi_resolution_spectral_loss'] \
            + losses['normalized_echo_density_loss'] \
            * self.config.model_config.normalized_echo_density_loss_beta
        return losses

    def forward(self, input_data):
        outputs = self.nn(
            x=input_data,
            early_rir=input_data.squeeze(1),
            h0=input_data[..., :self.config.model_config.rir_direct_length]
        )
        rir_generated = outputs[0]
        misc_param_exported = outputs[-1]
        return rir_generated.unsqueeze(1), misc_param_exported

    @staticmethod
    def multi_resolution_spectral_loss(rir_ground_truth, rir_generated, spec_function_list):
        n_type = len(spec_function_list)
        loss = 0
        for idx in range(n_type):
            spec_fn = spec_function_list[idx]
            spec_gt = 10 * (spec_fn(rir_ground_truth) + 1e-7).log10()
            spec_gen = 10 * (spec_fn(rir_generated) + 1e-7).log10()
            loss += torch.nn.L1Loss()(spec_gt, spec_gen)
        return loss

    @staticmethod
    def normalized_echo_density_loss(rir_ground_truth, rir_generated, **kwargs):
        rir_ground_truth = hpf_before_get_echo_density(rir_ground_truth).squeeze(-2)
        rir_generated = hpf_before_get_echo_density(rir_generated).squeeze(-2)
        ned_approx_gt = differentiable_ned(rir_ground_truth, **kwargs)
        ned_approx_gen = differentiable_ned(rir_generated, **kwargs)
        loss = (
            torch.nn.MSELoss(reduction='sum')(ned_approx_gt, ned_approx_gen)
            / (torch.ones_like(ned_approx_gt).sum() + 1e-8)
        ).sqrt()
        return loss

    @staticmethod
    def diffuse_insensitive_ned_error(rir_ground_truth, rir_generated, **kwargs):
        rir_ground_truth = hpf_before_get_echo_density(rir_ground_truth).squeeze(-2)
        rir_generated = hpf_before_get_echo_density(rir_generated).squeeze(-2)
        ned_gt = normalized_echo_density(rir_ground_truth, **kwargs)
        ned_gen = normalized_echo_density(rir_generated, **kwargs)
        fn = model.base.DIFFUSE_INSENSITIVE_FN
        error = (
            torch.nn.MSELoss(reduction='sum')(fn(ned_gt), fn(ned_gen))
            / (torch.ones_like(ned_gt).sum() + 1e-8)
        ).sqrt()
        return error
