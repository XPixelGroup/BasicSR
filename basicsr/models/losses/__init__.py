from .losses import (CharbonnierLoss, GANLoss, GradientPenaltyLoss, L1Loss,
                     MSELoss, PerceptualLoss, WeightedTVLoss)

__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'WeightedTVLoss', 'PerceptualLoss',
    'GANLoss', 'GradientPenaltyLoss'
]
