from src.config.default import _CN as cfg

# pose estimation
cfg.TRAINER.POSE_ESTIMATION_METHOD = 'LO-RANSAC'
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.EPI_ERR_THR = 1e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)

cfg.JAMMA.MP = False
cfg.JAMMA.EVAL_TIMES = 1
cfg.JAMMA.MATCH_COARSE.INFERENCE = True
cfg.JAMMA.FINE.INFERENCE = True
cfg.JAMMA.MATCH_COARSE.USE_SM = True
cfg.JAMMA.MATCH_COARSE.THR = 0.2
cfg.JAMMA.FINE.THR = 0.1

cfg.JAMMA.MATCH_COARSE.BORDER_RM = 2
