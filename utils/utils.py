import yaml
import os
import numpy as np
import random
import torch

def read_cfg(path: str=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg