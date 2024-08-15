import torch
import torchaudio

from model.base import BaseModule
from model.layers import Conv2dWithInitialization


class Encoder(BaseModule):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        spec_input = torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=1024, hop_length=128)
        self.spectral_embedding = lambda x: 10 * (spec_input.to(x.device)(x) + 1e-7).log10()

        self.out_sizes = self.config.model_config.encoder_out_channels
        self.in_sizes = [1] + self.out_sizes[:-1]
        self.kernel_sizes = [tuple(kernel_size) for kernel_size in self.config.model_config.encoder_kernel_sizes]
        self.strides = [tuple(stride) for stride in self.config.model_config.encoder_strides]
        self.paddings = [tuple(padding) for padding in self.config.model_config.encoder_paddings]
        self.conv_blocks = torch.nn.ModuleList(
            [
                EncoderConvBlock(
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

        self.fgru = FGRU(config=self.config)
        self.tgru = TGRU(config=self.config)
        self.bi_directional_tgru = 2 if self.config.model_config.encoder_tgru_bidirectional else 1
        self.after_gru_linear_dim = self.config.model_config.encoder_tgru_channel * self.bi_directional_tgru

        self.linear_1 = torch.nn.Linear(self.after_gru_linear_dim, self.after_gru_linear_dim)
        self.layer_norm_1 = torch.nn.LayerNorm(self.after_gru_linear_dim)
        self.linear_2 = torch.nn.Linear(self.after_gru_linear_dim, self.after_gru_linear_dim)
        self.layer_norm_2 = torch.nn.LayerNorm(self.after_gru_linear_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.spectral_embedding(x)

        for conv_block in self.conv_blocks:
            x = conv_block(x)
        assert \
            x.size(-2) == self.config.model_config.encoder_after_conv_frequency \
            and x.size(-1) == self.config.model_config.encoder_after_conv_time, \
            f"The shape of the tensor after encoder convolution layers should be [B, C, " \
            f"{self.config.model_config.encoder_after_conv_frequency}, " \
            f"{self.config.model_config.encoder_after_conv_time}], not [B, C, {x.size(-2)}, {x.size(-1)}]"

        x = self.fgru(x)
        x = self.tgru(x)

        x = self.linear_1(x)
        x = self.layer_norm_1(x)
        x = self.relu(x)

        x = self.linear_2(x)
        x = self.layer_norm_2(x)
        x = self.relu(x)
        return x


class EncoderConvBlock(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EncoderConvBlock, self).__init__()
        assert type(kernel_size) is tuple, \
            f"The data type of kernel_size should be tuple, not {type(kernel_size)}."
        assert type(stride) is tuple, \
            f"The data type of stride should be tuple, not {type(stride)}."
        assert type(padding) is tuple, \
            f"The data type of padding should be tuple, not {type(padding)}."

        self.convolution = Conv2dWithInitialization(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=(in_channels==1)
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        outputs = self.convolution(x)
        outputs = self.relu(outputs)
        return outputs


class FGRU(BaseModule):
    def __init__(self, config):
        super(FGRU, self).__init__()
        self.config = config
        self.fgru = torch.nn.GRU(
            input_size=self.config.model_config.encoder_out_channels[-1],
            hidden_size=self.config.model_config.encoder_fgru_channel,
            num_layers=self.config.model_config.encoder_fgru_num_layers,
            batch_first=True,
            bidirectional=self.config.model_config.encoder_fgru_bidirectional
        )

    def forward(self, x):
        assert len(x.size()) == 4, f"input must have 4 dimensions, got {len(x.size())}."
        gru_input = torch.transpose(x, 1, 3)

        gru_input = gru_input.contiguous().view(
            gru_input.size(0) * gru_input.size(1),
            gru_input.size(2),
            gru_input.size(3)
        )

        gru_output, _ = self.fgru(gru_input)
        gru_output = gru_output.view(x.size(0), x.size(3), x.size(2), -1)

        return gru_output


class TGRU(BaseModule):
    def __init__(self, config):
        super(TGRU, self).__init__()
        self.config = config
        self.bi_directional_fgru = 2 if self.config.model_config.encoder_fgru_bidirectional else 1
        self.tgru = torch.nn.GRU(
            input_size=self.config.model_config.encoder_fgru_channel 
            * self.bi_directional_fgru 
            * self.config.model_config.encoder_after_conv_frequency,
            hidden_size=self.config.model_config.encoder_tgru_channel,
            num_layers=self.config.model_config.encoder_tgru_num_layers,
            batch_first=True,
            bidirectional=self.config.model_config.encoder_tgru_bidirectional
        )

    def forward(self, x):
        assert len(x.size()) == 4, f"input must have 4 dimensions, got {len(x.size())}."
        gru_input = x.contiguous().view(x.size(0), x.size(1), -1)

        gru_output, _ = self.tgru(gru_input)

        return gru_output
