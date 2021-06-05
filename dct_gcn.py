"""
Neural networks for motion prediction.

Copyright ETH Zurich, Manuel Kaufmann
"""
from utils import get_dct_matrix
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from data import AMASSBatch
from losses import mse
from configuration import CONSTANTS as C
from gcn import GCN



class BaseModel(nn.Module):
    """A base class for neural networks that defines an interface and implements a few common functions."""

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.pose_size = config.pose_size
        self.create_model()
        self.is_test = False

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        """Create the model, called automatically by the initializer."""
        raise NotImplementedError("Must be implemented by subclass.")

    def forward(self, batch: AMASSBatch):
        """The forward pass."""
        raise NotImplementedError("Must be implemented by subclass.")

    def backward(self, batch: AMASSBatch, model_out):
        """The backward pass."""
        raise NotImplementedError("Must be implemented by subclass.")

    def model_name(self):
        """A summary string of this model. Override this if desired."""
        return '{}-lr{}'.format(self.__class__.__name__, self.config.lr)


class DCT_GCN(BaseModel):
    """
    This is a Graph Convolutional Network models the dynamics of dependent
    timeseries. The timeseries are first transformed into DCT representation.
    Implementation Adapted from: https://github.com/wei-mao-2019/LearnTrajDep 
    """

    def __init__(self, config):
        self.dropout = 0
        self.seed_seq_len = config.seed_seq_len
        self.target_seq_len = config.target_seq_len
        self.input_size = config.pose_size

        # Compute DCT matrices once
        dct_mat, idct_mat = get_dct_matrix(self.seed_seq_len + self.target_seq_len)
        self.dct_mat = Variable(torch.from_numpy(dct_mat)).float().to(C.DEVICE)
        self.idct_mat = Variable(torch.from_numpy(idct_mat)).float().to(C.DEVICE)

        # number of dct frequencies
        self.n_dct_freq = 20

        self.gcn_hidden_feature = 256  # Mao19 default param
        self.gcn_p_dropout      = 0.5  # Mao19 default param
        self.gcn_num_stage      = 12   # Mao19 default param

        super(DCT_GCN, self).__init__(config)

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        self.gcn = GCN(input_feature=self.n_dct_freq,
                        hidden_feature=self.gcn_hidden_feature, 
                        p_dropout=self.gcn_p_dropout,
                        num_stage=self.gcn_num_stage,
                        node_n=self.input_size) # N_JOINTS * DOF

    def forward(self, batch: AMASSBatch):
        """
        The forward pass.
        :param batch: Current batch of data.
        :return: Each forward pass must return a dictionary with keys {'seed', 'predictions'}.
        """
        model_out = {'seed': batch.poses[:, :self.config.seed_seq_len],
                     'predictions': None}

        batch_size = batch.batch_size

        ######################

        input_series = batch.poses[:, :self.seed_seq_len, :]
        # => (batchsize, self.seed_seq_len, N_JOINT * DOF)

        # prepare padding of input series
        # TODO: indices are allowed to by np.ndarrays?
        all_indices = np.arange(0, self.seed_seq_len)
        last_indeces = np.full(self.target_seq_len, self.seed_seq_len-1)
        index_padded = np.append(all_indices, last_indeces)
        # => (self.seed_seq_len + self.target_seq_len)

        # transform padded series to frequency domain
        input_dct = torch.matmul(self.dct_mat[:self.n_dct_freq,:], input_series[:, index_padded, :])
        # => (batchsize, self.n_dct_freq, N_JOINT * DOF)

        ######################

        # predict batchwise
        pred = self.gcn(input_dct.transpose(1,2))
        output_dct = pred.transpose(1,2)
        # => (batchsize, self.n_dct_freq, N_JOINT * DOF)


        ######################

        # transform back to time domain
        # TODO: Mao uses complicated transposes and stuff?
        output_series = torch.matmul(self.idct_mat[:,:self.n_dct_freq], output_dct)
        # => (batchsize, self.seed_seq_len + self.target_seq_len, N_JOINT * DOF)

        # store predictions back
        # TODO: Mao computes MSE over entire sequence,
        #       but template assumes predictions to be of length 24?
        if self.training:
            model_out['predictions'] = output_series
        else:
            model_out['predictions'] = output_series[:, self.config.seed_seq_len:, :]

        return model_out

    def backward(self, batch: AMASSBatch, model_out):
        """
        The backward pass.
        :param batch: The same batch of data that was passed into the forward pass.
        :param model_out: Whatever the forward pass returned.
        :return: The loss values for book-keeping, as well as the targets for convenience.
        """
        predictions = model_out['predictions']

        if self.training:
            targets = batch.poses
        else:
            targets = batch.poses[:, self.config.seed_seq_len:, :]

        total_loss = mse(predictions, targets)

        # If you have more than just one loss, just add them to this dict and they will automatically be logged.
        loss_vals = {'total_loss': total_loss.cpu().item()}

        if self.training:
            # We only want to do backpropagation in training mode, as this function might also be called when evaluating
            # the model on the validation set.
            total_loss.backward()

        return loss_vals, targets
