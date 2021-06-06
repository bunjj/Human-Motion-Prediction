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

class RNN2(BaseModel):
    """
    This models the implementation of RNN as described in
    simple clean code version
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

        super(RNN2, self).__init__(config)

    # noinspection PyAttributeOutsideInit
    def create_model(self):

        self.linear = nn.Linear(in_features=self.pose_size, out_features=self.linear_size)
        self.cell = nn.LSTMCell(input_size=self.linear_size, hidden_size=self.rnn_size)
        self.linear_pred_1 = nn.Linear(in_features=self.rnn_size, out_features=960)
        self.relu = nn.ReLU()
        self.linear_pred_2 = nn.Linear(in_features=960, out_features=self.pose_size)

        pass

    def forward(self, batch: AMASSBatch):
        """
        The forward pass.
        :param batch: Current batch of data.
        :return: Each forward pass must return a dictionary with keys {'seed', 'predictions'}.
        """
        def loop_function(prev, i):
            return prev

        model_out = {'seed': batch.poses[:, :self.config.seed_seq_len],
                     'predictions': None}

        batch_size = batch.batch_size

        prediction_inputs = batch.poses
        prediction_inputs = torch.transpose(prediction_inputs, 0, 1)

        state_h = torch.zeros(batch_size, self.rnn_size, device=C.DEVICE)
        state_c = torch.zeros(batch_size, self.rnn_size, device=C.DEVICE)

        all_outputs = []
        outputs = []
        prev = None 
        
        for i in range((self.seed_seq_len + self.target_seq_len-1)):
            
            if i < self.seed_seq_len or self.training:
                inp = prediction_inputs[i]
            else:
                inp = prev
                inp = inp.detach()
                
            state = self.linear(nn.functional.dropout(inp, self.dropout, training=self.training))
            (state_h, state_c) = self.cell(state, (state_h, state_c))

            state = self.linear_pred_1(state_h)
            state = self.relu(state)
            state = self.linear_pred_2(state)
            output = inp + state

            all_outputs.append(output.view([1, batch_size, self.pose_size]))
            if i >= (self.seed_seq_len-1):
                outputs.append(output.view([1, batch_size, self.pose_size]))

            prev = output


        outputs = torch.cat(outputs, 0)
        outputs = torch.transpose(outputs, 0, 1)

        all_outputs = torch.cat(all_outputs,0)
        all_outputs = torch.transpose(all_outputs, 0,1)

        model_out['predictions'] = outputs
        model_out['training_predictions'] = all_outputs

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
        #print("predictions " + str(predictions.shape))
        #print("targets " + str(targets.shape))
        #total_loss = mse(predictions, targets)

        all_predictions = model_out['training_predictions']
        all_targets = batch.poses[:, 1:]
        #print("all_predictions " + str(all_predictions.shape))
        #print("all_targets " + str(all_targets.shape))

        total_loss = loss_pose_joint_sum(all_predictions, all_targets)

        # If you have more than just one loss, just add them to this dict and they will automatically be logged.
        loss_vals = {'total_loss': total_loss.cpu().item()}

        if self.training:
            # We only want to do backpropagation in training mode, as this function might also be called when evaluating
            # the model on the validation set.
            total_loss.backward()

        return loss_vals, targets
