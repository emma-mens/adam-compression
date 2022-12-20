from torchpack.mtpack.utils.config import Config, configs

configs.train.compression.warmup_epochs = 7
configs.train.compression.snr_compression = True
configs.train.compression.snr_warmup = True
configs.train.compression.beta1 = 0.1
configs.train.compression.beta2 = 0.995
