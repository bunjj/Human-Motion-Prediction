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


def loss_pose_all_mean(predictions, targets):
    """
    Loss computed as described https://github.com/eth-ait/spl/blob/6b37cc0a61c69b6e43187800d6589eb9cfaa9799/spl/model/base_model.py
    """
    diff = predictions - targets
    return (diff * diff).mean()


def loss_pose_joint_sum(predictions, targets):
    """
    Loss computed as described https://github.com/eth-ait/spl/blob/6b37cc0a61c69b6e43187800d6589eb9cfaa9799/spl/model/base_model.py
    """
    diff = predictions - targets
    per_joint_loss = (diff * diff).view(-1, 143, 15, 9) #TODO: adjust such that it doesn't use hardcoded
    per_joint_loss = per_joint_loss.sum(dim=-1)
    per_joint_loss = per_joint_loss.sqrt()
    per_joint_loss = per_joint_loss.sum(dim=-1)

    return per_joint_loss.mean()


def mse_joint(predictions, targets):
    """
    Compute the MSE.
    :param predictions: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :param targets: A tensor of shape (N, MAX_SEQ_LEN, -1)
    :return: The MSE between predictions and targets.
    """
    diff = predictions - targets
    loss_per_sample_and_seq = (diff * diff).sum(dim=-1)  # (N, F)
    return loss_per_sample_and_seq.sum()