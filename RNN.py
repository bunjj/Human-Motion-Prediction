"""
Neural networks for motion prediction.

Copyright ETH Zurich, Manuel Kaufmann
"""
import torch
import torch.nn as nn

from data import AMASSBatch
from losses import *
from configuration import CONSTANTS as C
import numpy as np


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


class RNN(BaseModel):
    """
    This models the implementation of RNN as described in
    overly complicated but works by now
    https://ait.ethz.ch/projects/2019/spl/
    """

    def __init__(self, config):
        self.rnn_size = 1024
        self.dropout = 0.1
        self.spl_size = 128
        self.linear_size = 256
        self.seed_seq_len = config.seed_seq_len
        self.target_seq_len = config.target_seq_len
        self.pose_size = config.pose_size

        super(RNN, self).__init__(config)

    # noinspection PyAttributeOutsideInit
    def create_model(self):
        self.linear = nn.Linear(self.pose_size, self.linear_size)
        self.cell = nn.LSTMCell(self.linear_size, self.rnn_size)

        # This part corresponds to the SPL layer for the vanilla RNN model
        self.linear_r1 = nn.Linear(self.rnn_size, 960)
        self.relu = nn.ReLU()
        self.linear_r2 = nn.Linear(960, self.pose_size)

    def forward(self, batch: AMASSBatch):
        """
        The forward pass.
        :param batch: Current batch of data.
        :return: Each forward pass must return a dictionary with keys {'seed', 'predictions'}.
        """
        model_out = {'seed': batch.poses[:, :self.config.seed_seq_len],
                     'predictions': None}

        def loop_function(prev, i):
            return prev

        batch_size = batch.batch_size

        ######################

        encoder_inputs = batch.poses[:, 0:self.seed_seq_len, :]
        if not self.training:
            decoder_inputs = torch.zeros((batch.poses.shape[0], self.target_seq_len, batch.poses.shape[2]))
            decoder_inputs = decoder_inputs.to(C.DEVICE)
        else:
            decoder_inputs = batch.poses[:, self.seed_seq_len :, :]

        encoder_inputs = torch.transpose(encoder_inputs, 0, 1)
        decoder_inputs = torch.transpose(decoder_inputs, 0, 1)

        all_outputs = []
        state_h = torch.zeros(batch_size, self.rnn_size, device=C.DEVICE)
        state_c = torch.zeros(batch_size, self.rnn_size, device=C.DEVICE)
        for i in range(self.seed_seq_len):
            state = encoder_inputs[i]
            state = self.linear(nn.functional.dropout(state, self.dropout, training=self.training))
            state_h, state_c = self.cell(state, (state_h, state_c))
            state = self.linear_r1(state_h)
            state = self.relu(state)
            state = self.linear_r2(state)
            state = state + encoder_inputs[i]
            state = state.to(C.DEVICE)
            all_outputs.append(state.view([1, batch_size, self.pose_size]))

        outputs = []
        outputs.append(state.view([1, batch_size, self.pose_size]))

        prev = None
        for i, inp in enumerate(decoder_inputs[:self.target_seq_len-1]):
            if self.training:
                state = decoder_inputs[i]
            state = self.linear(nn.functional.dropout(state, self.dropout, training=self.training))
            state_h, state_c = self.cell(state, (state_h, state_c))
            state = self.linear_r1(state_h)
            state = self.relu(state)
            state = self.linear_r2(state)
            state = state + decoder_inputs[i]
            # state = state.to(C.DEVICE)
            outputs.append(state.view([1, batch_size, self.pose_size]))
            all_outputs.append(state.view([1, batch_size, self.pose_size]))

        outputs = torch.cat(outputs, 0)
        outputs = torch.transpose(outputs, 0, 1)
        all_outputs = torch.cat(all_outputs,0)
        all_outputs = torch.transpose(all_outputs, 0,1)
        model_out['predictions'] = outputs
        model_out['training_pred'] = all_outputs
        ##############################################
        # previous shizzle
        ##################3
        # model_in = batch.poses[:, self.config.seed_seq_len-self.n_history:self.config.seed_seq_len]
        # pred = self.dense(model_in.reshape(batch_size, -1))
        # model_out['predictions'] = pred.reshape(batch_size, self.config.target_seq_len, -1)
        ########################################
        return model_out

    def backward(self, batch: AMASSBatch, model_out):
        """
        The backward pass.
        :param batch: The same batch of data that was passed into the forward pass.
        :param model_out: Whatever the forward pass returned.
        :return: The loss values for book-keeping, as well as the targets for convenience.
        """
        predictions = model_out['predictions']
        targets = batch.poses[:, self.config.seed_seq_len:]

        all_predictions = model_out['training_pred']
        all_targets = batch.poses[:, 1:]
        # print('prediction shape:' + str(predictions[0].shape))
        # print('targets shape:' + str(targets[0].shape))
        # print('preall_prediction shape:' +  str(all_predictions[0].shape))
        # print('all_targets shape:' + str(all_targets[0].shape))
        total_loss = mse(all_predictions, all_targets)

        # If you have more than just one loss, just add them to this dict and they will automatically be logged.
        loss_vals = {'total_loss': total_loss.cpu().item()}

        if self.training:
            # We only want to do backpropagation in training mode, as this function might also be called when evaluating
            # the model on the validation set.
            total_loss.backward()

        return loss_vals, targets
