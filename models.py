"""
Neural networks for motion prediction.

Copyright ETH Zurich, Manuel Kaufmann
"""
import torch
import torch.nn as nn
from torch.nn.functional import max_pool1d

from data import AMASSBatch
from losses import mse


def create_model(config):
    """
    This is a helper function that can be useful if you have several model definitions that you want to
    choose from via the command line.
    """
    if config.model == 'ZeroVelocity':
        from ZeroVelocity import ZeroVelocity
        return ZeroVelocity(config)
    elif config.model == 'seq2seq':
        from Seq2Seq import Seq2Seq
        return Seq2Seq(config)
    elif config.model == 'seq2seq_lstm1':
        from Seq2Seq_LSTM1 import Seq2Seq_LSTM1
        return Seq2Seq_LSTM1(config)
    elif config.model == 'seq2seq_lstm2':
        from Seq2Seq_LSTM2 import Seq2Seq_LSTM2
        return Seq2Seq_LSTM2(config)
    elif config.model == 'seq2seq_lstm3':
        from Seq2Seq_LSTM3 import Seq2Seq_LSTM3
        return Seq2Seq_LSTM3(config)
    elif config.model == 'rnn':
        from RNN import RNN
        return RNN(config)
    elif config.model == 'rnn2':
        from RNN2 import RNN2
        return RNN2(config)
    elif config.model == 'rnn3':
        from RNN3 import RNN3
        return RNN3(config)
    elif config.model == 'spl':
        from SPL import SPL
        return SPL(config)
    elif config.model == 'dct_gcn':
        from dct_gcn import DCT_GCN
        return DCT_GCN(config)
    elif config.model == 'dct_att_gcn':
        from dct_att_gcn import DCT_ATT_GCN
        return DCT_ATT_GCN(config)
    else:
        from DummyModel import DummyModel
        return DummyModel(config)
