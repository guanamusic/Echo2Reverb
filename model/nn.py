from model.base import BaseModule
from model.echo_encoder import EchoEncoder
from model.encoder import Encoder
from model.projection_layer import ProjectionLayer
from model.reverberation_generator import EchoAwareReverberation


class Echo2ReverbNN(BaseModule):
    def __init__(self, config):
        super(Echo2ReverbNN, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.echo_encoder = EchoEncoder(config) if config.model_config.use_echo_encoder else None
        self.projection_layer = ProjectionLayer(config)
        self.reverberation_generator = EchoAwareReverberation(config)

    def forward(self, x, early_rir=None, h0=None):
        """
        Compute forward pass of neural network
        """
        outputs = self.reverberation_generator(
            rgp=self.projection_layer(
                spectral_latent=self.encoder(x),
                echo_latent=self.echo_encoder(x) if self.config.model_config.use_echo_encoder else None
            ),
            early_rir=early_rir,
            h0=h0
        )
        return outputs
