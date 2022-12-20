from torchpack.mtpack.utils.config import Config, configs

configs.train.compression.warmup_epochs = 7
configs.train.compression.snr_compression = True
configs.train.compression.snr_warmup = True
configs.train.compression.beta1 = 0.9
configs.train.compression.beta2 = 0.995
configs.train.compression.beta1_warmup = False 
configs.train.compression.l_t = 64 
configs.train.compression.bin_multiplier = 64
