import math

import torch

import model.base
from model.base import BaseModule
from tools import velvet_noise_generation, get_filterbank


class EchoAwareReverberation(BaseModule):
    def __init__(self, config):
        super(EchoAwareReverberation, self).__init__()
        sample_rate = config.data_config.sample_rate

        self.config = config
        self.rir_offset = config.model_config.rir_early_length
        self.rir_total_samples = int(config.data_config.rir_length_in_ms * sample_rate / 1000)

        # Initializing reverberation generator
        self.noise_type = config.model_config.noise_type
        assert self.noise_type in model.base.ALLOWED_NOISE.keys(), \
            f"noise_type: '{self.noise_type}' is not yet supported."
        self.noise_segment_length = config.model_config.noise_segment_length
        self.num_noise_segment = config.model_config.num_noise_segment

        # Initializing filterbank
        self.synthesis_filter_length_in_segment = config.model_config.synthesis_filter_length_in_segment
        self.synthesis_filter_stride_in_segment = config.model_config.synthesis_filter_stride_in_segment
        self.synthesis_filter_length = self.noise_segment_length * self.synthesis_filter_length_in_segment
        self.synthesis_filter_stride = self.noise_segment_length * self.synthesis_filter_stride_in_segment

        assert config.model_config.synthesis_filterbank_type in model.base.ALLOWED_SYNTHESIS_FILTERBANK_TYPE, \
            f"synthesis_filterbank_type: '{config.model_config.synthesis_filterbank_type}' is not yet supported."
        filterbank = get_filterbank(
            filterbank_type=config.model_config.synthesis_filterbank_type,
            sample_rate=config.data_config.sample_rate,
            signal_length=self.synthesis_filter_length,
            synthesis=True
        ) if config.model_config.synthesis_filterbank_type != "none" else torch.ones(self.synthesis_filter_length, 1)
        self.register_buffer(f"synthesis_filterbank", filterbank)

        synthesis_filter_window_type = config.model_config.synthesis_filter_window_type
        assert synthesis_filter_window_type in model.base.ALLOWED_SYNTHESIS_WINDOW.keys(), \
            f"synthesis_filter_window_type: '{synthesis_filter_window_type}' is not yet supported."
        self.register_buffer(
            f"synthesis_window",
            model.base.ALLOWED_SYNTHESIS_WINDOW[synthesis_filter_window_type](self.synthesis_filter_length)
        )

    def forward(self, rgp, early_rir=None, h0=None):
        assert h0 is not None
        filterbank_weight = rgp['filterbank_weight']
        filterbank = self.__getattr__(f"synthesis_filterbank")

        # Generating the noise segment
        noise_segment = self.generate_noise_segment(rgp, h0)

        # Preparing to coloration
        zero_pad_len_for_synthesis_filter = \
            self.synthesis_filter_length_in_segment - self.synthesis_filter_stride_in_segment
        zero_pad_len_additional = \
            math.ceil(self.num_noise_segment / self.synthesis_filter_length_in_segment) \
            * self.synthesis_filter_length_in_segment \
            - self.num_noise_segment
        zero_pad_len_front = zero_pad_len_for_synthesis_filter
        zero_pad_len_rear = zero_pad_len_for_synthesis_filter + zero_pad_len_additional
        noise_segment_reshape = torch.cat(
            (
                torch.zeros_like(noise_segment)[..., 0:zero_pad_len_front, :],
                noise_segment,
                torch.zeros_like(noise_segment)[..., 0:zero_pad_len_rear, :]
            ),
            dim=-2
        )
        synthesis_window_noise = noise_segment_reshape.unfold(
            -2,
            self.synthesis_filter_length_in_segment,
            self.synthesis_filter_stride_in_segment
        ).transpose(-1, -2).view(noise_segment.size(0), -1, self.synthesis_filter_length)

        # Zero-phase filtering
        assert filterbank_weight.size(-2) == synthesis_window_noise.size(-2)
        zero_phase_filter = (
            filterbank.transpose(-1, -2).unsqueeze(0).unsqueeze(0).expand(filterbank_weight.size()[:2] + (-1, -1))
            * filterbank_weight.unsqueeze(-1)
        ).sum(-2)
        n_fft = synthesis_window_noise.size(-1)
        filtered_noise = torch.fft.ifft(
            torch.fft.fft(synthesis_window_noise, n=n_fft, dim=-1) * zero_phase_filter,
            n=n_fft,
            dim=-1
        ).real * self.__getattr__(f"synthesis_window")

        # Folding (overlap-add)
        filtered_noise = filtered_noise.transpose(-2, -1)
        filtered_noise = torch.nn.functional.fold(
            input=filtered_noise,
            output_size=(1, self.noise_segment_length * filtered_noise.size(-1) + self.synthesis_filter_length - 1),
            kernel_size=(1, self.synthesis_filter_length),
            stride=(1, self.synthesis_filter_stride)
        ).squeeze(-2).squeeze(-2)[..., self.synthesis_filter_length - self.synthesis_filter_stride:]

        # Concat
        h = torch.nn.functional.pad(
            input=early_rir,
            pad=(0, self.rir_total_samples - self.rir_offset),
            mode='constant'
        )
        filtered_noise = torch.nn.functional.pad(
            input=filtered_noise,
            pad=(self.rir_offset, self.rir_total_samples - self.rir_offset - filtered_noise.size(-1)),
            mode='constant'
        )
        h = h + filtered_noise

        misc = None
        try:
            if self.export_misc:
                misc = {f"{key}": value for key, value in rgp.items()}
                misc['noise_segment'] = noise_segment
                misc['synthesis_filterbank'] = filterbank
                misc['zero_phase_filter'] = zero_phase_filter
        except:
            pass

        return h, misc

    @staticmethod
    def gaussian_velvet_sampling(velvet_log_gain, gaussian_log_gain, signal_length, h0):
        # Converting direct response to be zero-mean
        h0 = h0 - h0.mean(-1, keepdim=True)
        # Generating interleaved velvet noise to avoid overlap by direct response
        velvet = velvet_noise_generation(
            average_pulse_distance=signal_length // 3,
            interleave_position=h0.size(-1),
            pulse_log_gain=velvet_log_gain,
            shape_tuple=gaussian_log_gain.size()[:-1] + (signal_length,)
        )
        n_fft = signal_length + h0.size(-1) - 1
        # Convolution with h0
        velvet_h0 = torch.fft.ifft(
            torch.fft.fft(velvet, n=n_fft, dim=-1) * torch.fft.fft(h0, n=n_fft, dim=-1),
            n=n_fft,
            dim=-1
        ).real[..., :signal_length]
        gaussian = torch.randn_like(velvet_h0).detach()
        conv_velvet_gaussian = velvet_h0 + gaussian * torch.exp(gaussian_log_gain)
        return conv_velvet_gaussian

    def generate_noise_segment(self, rgp, h0):
        if self.noise_type == 'gaussian':
            gaussian_gain = rgp['gain']
            gaussian_sampled = \
                torch.randn(gaussian_gain.size()[:-1] + (self.noise_segment_length,)).to(rgp['gain'].device)
            noise_segment = gaussian_sampled.detach() * gaussian_gain
        elif self.noise_type == 'conv-velvet-gaussian':
            velvet_log_gain = rgp['velvet_log_gain']
            gaussian_log_gain = rgp['gaussian_log_gain']
            noise_segment = self.gaussian_velvet_sampling(
                velvet_log_gain=velvet_log_gain,
                gaussian_log_gain=gaussian_log_gain,
                signal_length=self.noise_segment_length,
                h0=h0
            )
        else:
            raise NotImplementedError
        return noise_segment
