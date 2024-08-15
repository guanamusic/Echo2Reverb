import torch

from model.base import BaseModule
from model.layers import Conv1dWithInitialization
from tools import hpf_before_get_echo_density


class EchoEncoder(BaseModule):
    def __init__(self, config):
        super(EchoEncoder, self).__init__()
        self.config = config

        self.echo_embedding_offset = config.model_config.rir_direct_length
        self.echo_embedding_segment_size = config.model_config.echo_embedding_segment_size
        self.echo_embedding_histogram_bins = config.model_config.echo_embedding_histogram_bins
        self.echo_embedding_stride = config.model_config.echo_embedding_stride

        self.out_sizes = config.model_config.echo_encoder_out_channels
        self.in_sizes = [self.echo_embedding_histogram_bins] + self.out_sizes[:-1]
        self.kernel_sizes = config.model_config.echo_encoder_kernel_sizes
        self.strides = config.model_config.echo_encoder_strides
        self.paddings = [(kernel_size - 1) // 2 for kernel_size in self.kernel_sizes]
        self.conv_blocks = torch.nn.ModuleList(
            [
                EchoEncoderConvBlock(
                    in_channels=in_size,
                    out_channels=out_size,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                ) for in_size, out_size, kernel_size, stride, padding in zip(
                    self.in_sizes,
                    self.out_sizes,
                    self.kernel_sizes,
                    self.strides,
                    self.paddings
                )
            ]
        )

        self.tgru = TGRU(config=self.config)
        self.bi_directional_tgru = 2 if self.config.model_config.echo_encoder_tgru_bidirectional else 1
        self.after_gru_linear_dim = self.config.model_config.encoder_tgru_channel * self.bi_directional_tgru

        self.linear_1 = torch.nn.Linear(self.after_gru_linear_dim, self.after_gru_linear_dim)
        self.layer_norm_1 = torch.nn.LayerNorm(self.after_gru_linear_dim)
        self.linear_2 = torch.nn.Linear(self.after_gru_linear_dim, self.after_gru_linear_dim)
        self.layer_norm_2 = torch.nn.LayerNorm(self.after_gru_linear_dim)
        self.relu = torch.nn.ReLU()

    @staticmethod
    def echo_embedding(x, offset=50, segment_size=750, histogram_bins=50, stride=250):
        assert len(x.size()) == 3

        # Trimming w/ padding
        x = x[..., offset:]
        if x.size(-1) % segment_size != 0:
            x = torch.nn.functional.pad(x, (0, segment_size - (x.size(-1) % segment_size)), mode='constant', value=0.)
        x = x.unfold(-1, segment_size, stride).squeeze(-3)

        # Normalizing w/ power & scaling
        x = x / (x.pow(2).sum(-1, keepdim=True) + 1e-9).sqrt()
        x = x.pow(2) / (x.pow(2).max(-1, keepdim=True)[0] + 1e-9) * (histogram_bins - 1e-4)

        # Histogram-ize
        x = torch.floor(x)
        x = torch.nn.functional.one_hot(x.to(torch.long), num_classes=histogram_bins).sum(-2) * 1. / segment_size

        x = x.transpose(-1, -2)
        return x

    def forward(self, x):
        x = self.echo_embedding(
            x=hpf_before_get_echo_density(x.squeeze(1), e2r_trigger=True),
            offset=self.echo_embedding_offset,
            segment_size=self.echo_embedding_segment_size,
            histogram_bins=self.echo_embedding_histogram_bins,
            stride=self.echo_embedding_stride
        )

        for conv_block in self.conv_blocks:
            x = conv_block(x)
        assert x.size(-1) == self.config.model_config.echo_encoder_after_conv_time, \
            f"The shape of the tensor after echo encoder convolution layers should be " \
            f"[B, C, {self.config.model_config.echo_encoder_after_conv_time}], not [B, C, {x.size(-1)}]"

        x = x.transpose(-1, -2)
        x = self.tgru(x)

        x = self.linear_1(x)
        x = self.layer_norm_1(x)
        x = self.relu(x)

        x = self.linear_2(x)
        x = self.layer_norm_2(x)
        x = self.relu(x)

        return x


class EchoEncoderConvBlock(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EchoEncoderConvBlock, self).__init__()
        self.convolution = Conv1dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(in_channels==1)
        )
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        outputs = self.convolution(x)
        outputs = self.leaky_relu(outputs)
        return outputs


class TGRU(BaseModule):
    def __init__(self, config):
        super(TGRU, self).__init__()
        self.config = config
        self.tgru = torch.nn.GRU(
            input_size=config.model_config.echo_encoder_out_channels[-1],
            hidden_size=self.config.model_config.echo_encoder_tgru_channel,
            num_layers=self.config.model_config.echo_encoder_tgru_num_layers,
            batch_first=True,
            bidirectional=self.config.model_config.echo_encoder_tgru_bidirectional
        )

    def forward(self, x):
        x, _ = self.tgru(x)

        return x
