from torchpack.mtpack.utils.config import Config, configs

configs.train.compression.warmup_epochs = 3
configs.train.compression.snr_compression = False 
configs.train.compression.snr_warmup = False
configs.train.compression.snr_init = "grad_init"
configs.train.compression.init_snr_after_warmup = True
configs.train.compression.use_bias_correction = True