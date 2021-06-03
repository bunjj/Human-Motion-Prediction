"""
Some utility functions.

Copyright ETH Zurich, Manuel Kaufmann
"""
import glob
import os
import zipfile
import numpy as np

def create_model_dir(experiment_main_dir, experiment_id, model_summary):
    """
    Create a new model directory.
    :param experiment_main_dir: Where all experiments are stored.
    :param experiment_id: The ID of this experiment.
    :param model_summary: A summary string of the model.
    :return: A directory where we can store model logs. Raises an exception if the model directory already exists.
    """
    model_name = "{}-{}".format(experiment_id, model_summary)
    model_dir = os.path.join(experiment_main_dir, model_name)
    if os.path.exists(model_dir):
        raise ValueError("Model directory already exists {}".format(model_dir))
    os.makedirs(model_dir)
    return model_dir


def get_model_dir(experiment_dir, model_id):
    """Return the directory in `experiment_dir` that contains the given `model_id` string."""
    model_dir = glob.glob(os.path.join(experiment_dir, str(model_id) + "-*"), recursive=False)
    return None if len(model_dir) == 0 else model_dir[0]


def export_code(file_list, output_file):
    """Stores files in a zip."""
    if not output_file.endswith('.zip'):
        output_file += '.zip'
    ofile = output_file
    counter = 0
    while os.path.exists(ofile):
        counter += 1
        ofile = output_file.replace('.zip', '_{}.zip'.format(counter))
    zipf = zipfile.ZipFile(ofile, mode="w", compression=zipfile.ZIP_DEFLATED)
    for f in file_list:
        zipf.write(f)
    zipf.close()


def count_parameters(net):
    """Count number of trainable parameters in `net`."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def get_dct_matrix(N):
    """
    Construct DCT and IDCT matrice for (inverse) discrete cosine transform.
    Adapted from https://github.com/wei-mao-2019/LearnTrajDep/blob/master/utils/data_utils.py#L781
    """
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m