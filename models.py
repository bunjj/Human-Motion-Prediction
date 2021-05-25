"""
Neural networks for motion prediction.

Copyright ETH Zurich, Manuel Kaufmann
"""
import torch
import torch.nn as nn

from data import AMASSBatch
from losses import mse


def create_model(config):
    # This is a helper function that can be useful if you have several model definitions that you want to
    # choose from via the command line. For now, we just return the Dummy model.
    if config.model == 'ZeroVelocity':
        from ZeroVelocity import ZeroVelocity
        return ZeroVelocity(config)
    elif config.model == 'seq2seq_ib':
        from Seq2Seq_ib import Seq2Seq
        return Seq2Seq(config)
    else:
        from DummyModel import DummyModel
        return DummyModel(config)
