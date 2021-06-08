"""
Evaluate a model on the test set.

Copyright ETH Zurich, Manuel Kaufmann
"""
import argparse
import numpy as np
import os
import pandas as pd
import torch
import collections
import time
import utils as U

from configuration import Configuration
from configuration import CONSTANTS as C
from data import AMASSBatch
from data import LMDBDataset
from data_transforms import ToTensor, LogMap
from fk import SMPLForwardKinematics
from models import create_model
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from visualize import Visualizer
from motion_metrics import MetricsEngine


def _export_results(eval_result, output_file):
    """
    Write predictions into a file that can be uploaded to the submission system.
    :param eval_result: A dictionary {sample_id => (prediction, seed)}
    :param output_file: Where to store the file.
    """

    def to_csv(fname, poses, ids, split=None):
        n_samples, seq_length, dof = poses.shape
        data_r = np.reshape(poses, [n_samples, seq_length * dof])
        cols = ['dof{}'.format(i) for i in range(seq_length * dof)]

        # add split id very last
        if split is not None:
            data_r = np.concatenate([data_r, split[..., np.newaxis]], axis=-1)
            cols.append("split")

        data_frame = pd.DataFrame(data_r,
                                  index=ids,
                                  columns=cols)
        data_frame.index.name = 'Id'

        if not fname.endswith('.gz'):
            fname += '.gz'

        data_frame.to_csv(fname, float_format='%.8f', compression='gzip')

    sample_file_ids = []
    sample_poses = []
    for k in eval_result:
        sample_file_ids.append(k)
        sample_poses.append(eval_result[k][0])

    to_csv(output_file, np.stack(sample_poses), sample_file_ids)


def load_model_weights(checkpoint_file, net, state_key='model_state_dict'):
    """Loads a pre-trained model."""
    if not os.path.exists(checkpoint_file):
        raise ValueError("Could not find model checkpoint {}.".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location=C.DEVICE)
    ckpt = checkpoint[state_key]
    net.load_state_dict(ckpt)

    iteration = checkpoint['iteration']
    epoch = checkpoint['epoch']
    return iteration, epoch


def get_model_config(model_dir):
    model_config = Configuration.from_json(os.path.join(model_dir, 'config.json'))
    return model_config


def load_model(model_dir):
    model_config = get_model_config(model_dir)
    net = create_model(model_config)

    net.to(C.DEVICE)
    print('Model created with {} trainable parameters and config \n{}'.format(
        U.count_parameters(net), model_config))

    # Load model weights.
    checkpoint_file = os.path.join(model_dir, 'model.pth')
    iteration, epoch = load_model_weights(checkpoint_file, net)
    print('Loaded weights from {}'.format(checkpoint_file))

    return net, model_config, model_dir, (iteration, epoch)


def _evaluate(net, data_loader, metrics_engine):
    """
    NOTE: this is a copy of the same function in train.py
    Evaluate a model on the given dataset. This computes the loss, but does not do any backpropagation or gradient
    update.
    :param net: The model to evaluate.
    :param data_loader: The dataset.
    :param metrics_engine: MetricsEngine to compute metrics.
    :return: The loss value.
    """
    # Put the model in evaluation mode.
    net.eval()

    # Some book-keeping.
    loss_vals_agg = collections.defaultdict(float)
    n_samples = 0
    metrics_engine.reset()

    with torch.no_grad():
        for abatch in data_loader:
            # Move data to GPU.
            batch_gpu = abatch.to_gpu()

            # Get the predictions.
            model_out = net(batch_gpu)

            # Compute the loss.
            loss_vals, targets = net.backward(batch_gpu, model_out)

            # Accumulate the loss and multiply with the batch size (because the last batch might have different size).
            for k in loss_vals:
                loss_vals_agg[k] += loss_vals[k] * batch_gpu.batch_size

            # Compute metrics.
            metrics_engine.compute_and_aggregate(model_out['predictions'], targets)

            n_samples += batch_gpu.batch_size

    # Compute the correct average for the entire data set.
    for k in loss_vals_agg:
        loss_vals_agg[k] /= n_samples

    return loss_vals_agg


def evaluate_test(model_dir, predict=True, viz=False):
    """
    Load a model, evaluate it on the test set and save the predictions into the model directory.
    :param model_dir: The directory of the model to load.
    :param viz: If some samples should be visualized.
    """
    assert os.path.isdir(model_dir), "model_dir is not a directory"
    net, model_config, model_dir, (epoch, iteration) = load_model(model_dir)

    # No need to extract windows for the test set, since it only contains the seed sequence anyway.
    if model_config.repr == "rotmat":
        valid_transform = transforms.Compose([ToTensor()])
        test_transform = transforms.Compose([ToTensor()])
    elif model_config.repr == "axangle":
        test_transform = transforms.Compose([LogMap(), ToTensor()])
        valid_transform = transforms.Compose([LogMap(), ToTensor()])
    else:
        raise ValueError(f"Unkown representation: {model_config.repr}")


    valid_data = LMDBDataset(os.path.join(C.DATA_DIR, "validation"), transform=valid_transform)
    valid_loader = DataLoader(valid_data,
                              batch_size=model_config.bs_eval,
                              shuffle=False,
                              num_workers=model_config.data_workers,
                              collate_fn=AMASSBatch.from_sample_list)
    
    test_data = LMDBDataset(os.path.join(C.DATA_DIR, "test"), transform=test_transform)
    test_loader = DataLoader(test_data,
                             batch_size=model_config.bs_eval,
                             shuffle=False,
                             num_workers=model_config.data_workers,
                             collate_fn=AMASSBatch.from_sample_list)
    
    # Evaluate on validation
    print('Evaluate model on validation set:')
    start = time.time()
    net.eval()
    me = MetricsEngine(C.METRIC_TARGET_LENGTHS, model_config.repr)
    valid_losses = _evaluate(net, valid_loader, me)
    valid_metrics = me.get_final_metrics()
    elapsed = time.time() - start
    
    loss_string = ' '.join(['{}: {:.6f}'.format(k, valid_losses[k]) for k in valid_losses])
    print('[VALID {:0>5d} | {:0>3d}] {} elapsed: {:.3f} secs'.format(
                    iteration + 1, epoch + 1, loss_string, elapsed))
    print('[VALID {:0>5d} | {:0>3d}] {}'.format(
                    iteration + 1, epoch + 1, me.get_summary_string(valid_metrics)))
    
    # add validation metrics to config
    model_config.update(me.to_dict(valid_metrics, 'valid'))
    model_config.to_json(os.path.join(model_dir, 'config.json'))


    if predict:
        # Put the model in evaluation mode.
        net.eval()
        net.is_test = True
        results = dict()
        with torch.no_grad():
            for abatch in test_loader:
                # Move data to GPU.
                batch_gpu = abatch.to_gpu()

                # Get the predictions.
                model_out = net(batch_gpu)

                for b in range(abatch.batch_size):

                    predictions = model_out['predictions'][b].detach().cpu().numpy()
                    seed = model_out['seed'][b].detach().cpu().numpy()

                    if model_config.repr == 'axangle':
                        predictions = U.axangle2rotmat(predictions)
                        seed = U.axangle2rotmat(seed)

                    results[batch_gpu.seq_ids[b]] = (predictions, seed)

        fname = 'predictions_in{}_out{}.csv'.format(model_config.seed_seq_len, model_config.target_seq_len)
        _export_results(results, os.path.join(model_dir, fname))

    if predict and viz:
        fk_engine = SMPLForwardKinematics()
        visualizer = Visualizer(fk_engine)
        n_samples_viz = 10
        rng = np.random.RandomState(42)
        idxs = rng.randint(0, len(results), size=n_samples_viz)
        sample_keys = [list(sorted(results.keys()))[i] for i in idxs]
        for k in sample_keys:
            visualizer.visualize(results[k][1], results[k][0], title='Sample ID: {}'.format(k))
    
    net.is_test = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', required=True, help='Which models to evaluate.')
    parser.add_argument('--no_predict', action='store_true', help='Do not compute predictions for test data.')
    parser.add_argument('--viz', action='store_true', help='Visualize results.')
    args = parser.parse_args()

    model_dirs = U.get_model_dirs(C.EXPERIMENT_DIR, args.model_id)

    for model_dir in model_dirs:
        print(f'Processing {model_dir}')
        evaluate_test(model_dir, predict=(not args.no_predict), viz=args.viz)

