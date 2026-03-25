import torch
import random
import numpy as np
import logging

def set_seed(seed=42):
    """Step 1: Set Random Seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(name="Essawy_DL"):
    """Step 4: Log Everything"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    if not logger.handlers:
        # Console output
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File output (elly kan na2es)
        fh = logging.FileHandler('training_progress.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
