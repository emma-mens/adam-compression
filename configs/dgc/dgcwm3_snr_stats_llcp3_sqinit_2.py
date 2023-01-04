from torchpack.mtpack.utils.config import Config, configs

configs.train.compression.warmup_epochs = 3
configs.train.compression.snr_compression = True 
configs.train.compression.snr_warmup = False
configs.train.compression.sq_init_factor = 2.0
configs.train.compression.snr_init = "av_grad_init"