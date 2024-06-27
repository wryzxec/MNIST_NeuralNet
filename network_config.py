DEFAULT_LAYER_ARCHITECTURE = [30, 20, 10]
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 128
DEFAULT_ALPHA = 0.5
DEFAULT_BETA = 0.9
DEFAULT_MOMENTUM_APPLIED = False

class NetworkConfig:
    def __init__(self,
                layer_architecture=DEFAULT_LAYER_ARCHITECTURE,
                epochs=DEFAULT_EPOCHS,
                batch_size=DEFAULT_BATCH_SIZE,
                alpha=DEFAULT_ALPHA,
                beta=DEFAULT_BETA,
                momentum_applied=DEFAULT_MOMENTUM_APPLIED):
        self.layer_architecture = layer_architecture
        self.epochs = epochs
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.momentum_applied = momentum_applied