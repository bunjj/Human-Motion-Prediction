"""
Neural networks for motion prediction.

Copyright ETH Zurich, Manuel Kaufmann
"""
import torch
import torch.nn as nn

from data import AMASSBatch
from losses import mse



class BaseModel(nn.Module):
    """A base class for neural networks that defines an interface and implements a few common functions."""

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.pose_size = config.pose_size
        self.create_model()
        self.is_test = False
        self.loss_fun = None

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


class Seq2Seq(BaseModel):
    """
    This is a  seq2seq model as implemented in https://github.com/enriccorona/human-motion-prediction-pytorch/blob/master/src/seq2seq_model.py .
    """

    def __init__(self, config):
        self.n_history = 10
        self.rnn_size = 1024
        self.dropout = 0
        self.seed_seq_len = config.seed_seq_len
        self.target_seq_len = config.target_seq_len
        self.input_size = config.pose_size

        self.use_cuda = torch.cuda.is_available()
        super(Seq2Seq, self).__init__(config)
        print(vars(self))

    def create_model(self):
        self.cell = nn.GRUCell(self.input_size, self.rnn_size)
        self.fc1 = nn.Linear(self.rnn_size, self.config.pose_size)

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

        encoder_inputs = batch.poses[:, 0:self.seed_seq_len - 1, :]
        if self.is_test:
            decoder_inputs = torch.zeros((batch.poses.shape[0], self.target_seq_len, batch.poses.shape[2]))
            decoder_inputs[:,0,:] = batch.poses[:,self.seed_seq_len-1, :]
            if self.use_cuda:
                decoder_inputs = decoder_inputs.cuda()
        else:
            decoder_inputs  = batch.poses[:, self.seed_seq_len-1:self.seed_seq_len+self.target_seq_len-1, :]
        
        
        encoder_inputs = torch.transpose(encoder_inputs, 0, 1)
        decoder_inputs = torch.transpose(decoder_inputs, 0, 1)

        state = torch.zeros(batch_size, self.rnn_size)

        if self.use_cuda:
            state = state.cuda()
        for i in range(self.seed_seq_len - 1):
            state = self.cell(encoder_inputs[i], state)
            state = nn.functional.dropout(state, self.dropout, training=self.training)
            if self.use_cuda:
                state = state.cuda()

        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                inp = loop_function(prev, i)

            inp = inp.detach()

            state = self.cell(inp, state)

            output = inp + self.fc1(nn.functional.dropout(state, self.dropout, training=self.training))
            outputs.append(output.view([1, batch_size, self.input_size]))

            if loop_function is not None:
                prev = output

        outputs = torch.cat(outputs, 0)
        outputs = torch.transpose(outputs, 0, 1)
        model_out['predictions'] = outputs

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

        total_loss = self.loss_fun(predictions, targets)

        # If you have more than just one loss, just add them to this dict and they will automatically be logged.
        loss_vals = {'total_loss': total_loss.cpu().item()}

        if self.training:
            # We only want to do backpropagation in training mode, as this function might also be called when evaluating
            # the model on the validation set.
            total_loss.backward()

        return loss_vals, targets
