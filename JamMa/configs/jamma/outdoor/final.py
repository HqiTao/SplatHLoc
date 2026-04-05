from src.config.default import _CN as cfg

cfg.JAMMA.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
cfg.JAMMA.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
cfg.JAMMA.COARSE.D_MODEL = 256
cfg.JAMMA.FINE.D_MODEL = 64

cfg.TRAINER.SCHEDULER = 'CosineAnnealing'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
cfg.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
cfg.TRAINER.CANONICAL_BS = 2
cfg.TRAINER.CANONICAL_LR = 1e-4
cfg.TRAINER.WARMUP_STEP = 58200  # 3 epochs

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.JAMMA.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
cfg.JAMMA.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 20
cfg.TRAINER.N_VAL_PAIRS_TO_PLOT = 1

cfg.TRAINER.SEED = 66