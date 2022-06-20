class Architecture:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def encoder_layers(self):
        raise NotImplementedError()

    def decoder_layers(self):
        raise NotImplementedError()