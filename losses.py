"""
Some loss functions.

Copyright ETH Zurich, Manuel Kaufmann
"""


def mse(predictions, targets):
    """
    Compute the MSE.
    :param predictions: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :param targets: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :return: The MSE between predictions and targets.
    """
    diff = predictions - targets
    loss_per_sample_and_seq = (diff * diff).sum(dim=-1)  # (N, F)
    return loss_per_sample_and_seq.mean()


def rmse(predictions, targets):
    """
    Compute the RMSE.
    :param predictions: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :param targets: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :return: The RMSE between predictions and targets.
    """
    diff = predictions - targets
    loss_per_sample_and_seq = (diff * diff).sum(dim=-1)  # (N, F)
    loss_per_sample_and_seq = loss_per_sample_and_seq.sqrt()
    return loss_per_sample_and_seq.mean()


def loss_pose_all_mean(predictions, targets):
    """
    Loss computed as described https://github.com/eth-ait/spl/blob/6b37cc0a61c69b6e43187800d6589eb9cfaa9799/spl/model/base_model.py
    """
    diff = predictions - targets
    return (diff * diff).mean()

def loss_pose_joint_sum(predictions, targets, n_frames=144):
    """
    Loss computed as described https://github.com/eth-ait/spl/blob/6b37cc0a61c69b6e43187800d6589eb9cfaa9799/spl/model/base_model.py
    """
    diff = predictions - targets
    per_joint_loss = (diff*diff).view(-1, n_frames, 15, 9)
    per_joint_loss = per_joint_loss.sum(dim=-1)
    per_joint_loss = per_joint_loss.sqrt()
    per_joint_loss = per_joint_loss.sum(dim=-1)
    
    return per_joint_loss.mean()
