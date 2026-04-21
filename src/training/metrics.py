from __future__ import annotations
import torch
import torch.nn.functional as F

# class indices
# 0: Background (ignored in aggregation)
# 1: No damage
# 2: Minor damage
# 3: Major damage
# 4: Destroyed