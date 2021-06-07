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

spl_size = 64
rnn_size = 1024

class SPL_joint(nn.Module):
    """ Function needed to define a layer that moddels the Layer needed in SPL
     assumes that matrix representation for angles is used"""

    def __init__(self, n_parents):
        super().__init__()
        self.linear_1 = nn.Linear(in_features= rnn_size + n_parents * 9, out_features= spl_size)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(in_features= spl_size, out_features= 9)

    def forward(self, x):
        state = self.linear_1(x)
        state = self.relu(state)
        state = self.linear_2(state)

        return state


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

class SPL(BaseModel):
    """
    This models the implementation of RNN as described in
    https://ait.ethz.ch/projects/2019/spl/
    """

    def __init__(self, config):
        self.rnn_size = rnn_size
        self.dropout = 0.1
        self.spl_size = spl_size
        self.linear_size = 256
        self.seed_seq_len = config.seed_seq_len
        self.target_seq_len = config.target_seq_len
        self.pose_size = config.pose_size

        super(SPL, self).__init__(config)

    # noinspection PyAttributeOutsideInit
    def create_model(self):

        self.linear = nn.Linear(in_features=self.pose_size, out_features=self.linear_size)
        self.cell = nn.LSTMCell(input_size=self.linear_size, hidden_size=self.rnn_size)
        self.l_hip = SPL_joint(0)
        self.r_hip = SPL_joint(0)
        self.spine1 = SPL_joint(0)
        self.l_knee = SPL_joint(1)
        self.r_knee = SPL_joint(1)
        self.spine2 = SPL_joint(1)
        self.spine3 = SPL_joint(2)
        self.neck = SPL_joint(3)
        self.l_collar = SPL_joint(3)
        self.r_collar = SPL_joint(3)
        self.head = SPL_joint(4)
        self.l_shoulder = SPL_joint(4)
        self.r_shoulder = SPL_joint(4)
        self.l_elbow = SPL_joint(5)
        self.r_elbow = SPL_joint(5)
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
            state = self.relu(state)
            (state_h, state_c) = self.cell(state, (state_h, state_c))
            l_hip =  self.l_hip(state_h)
            r_hip = self.r_hip(state_h)
            spine1 = self.spine1(state_h)
            l_knee = self.l_knee(torch.cat((state_h, l_hip), dim = -1))
            r_knee = self.r_knee(torch.cat((state_h, r_hip), dim = -1))
            spine2 = self.spine2(torch.cat((state_h,  spine1), dim = -1))
            spine3 = self.spine3(torch.cat((state_h,  spine1, spine2), dim = -1))
            neck = self.neck(torch.cat((state_h,  spine1, spine2, spine3), dim = -1))
            l_collar = self.l_collar(torch.cat((state_h,  spine1, spine2, spine3), dim = -1))
            r_collar = self.r_collar(torch.cat((state_h,  spine1, spine2, spine3), dim = -1))
            head = self.head(torch.cat((state_h,  spine1, spine2, spine3, neck), dim = -1))
            l_shoulder = self.l_shoulder(torch.cat((state_h,  spine1, spine2, spine3, l_collar), dim = -1))
            r_shoulder = self.r_shoulder(torch.cat((state_h,  spine1, spine2, spine3, r_collar), dim = -1))
            l_elbow = self.l_elbow(torch.cat((state_h,  spine1, spine2, spine3, l_collar, l_shoulder), dim = -1))
            r_elbow = self.r_elbow(torch.cat((state_h,  spine1, spine2, spine3,r_collar, r_shoulder), dim = -1))

            #print(torch.cat((l_hip, r_hip, spine1), dim = -1).shape)

            state = torch.cat((l_hip, r_hip, spine1, l_knee, r_knee, spine2, spine3, neck,
                               l_collar, r_collar, head, l_shoulder, r_shoulder, l_elbow, r_elbow), dim = -1)

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

        total_loss = loss_pose_joint_sum_squared(all_predictions, all_targets)
        #print(total_loss)

        # If you have more than just one loss, just add them to this dict and they will automatically be logged.
        loss_vals = {'total_loss': total_loss.cpu().item()}

        if self.training:
            # We only want to do backpropagation in training mode, as this function might also be called when evaluating
            # the model on the validation set.
            total_loss.backward()

        return loss_vals, targets
