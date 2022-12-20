from torchpack.mtpack.utils.config import Config, configs

configs.train.compression.warmup_epochs = 7
configs.train.compression.snr_compression = True
configs.train.compression.snr_warmup = False
configs.train.compression.snr_init = 'zeros'
