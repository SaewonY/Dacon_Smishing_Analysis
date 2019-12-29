import numpy as np
import random
import os
import torch


ON_KAGGLE = 'KAGGLE_WORKING_DIR' in os.environ


def seed_everything(seed=123):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True