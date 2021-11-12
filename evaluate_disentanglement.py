# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example script how to get started with research using disentanglement_lib.

To run the example, please change the working directory to the containing folder
and run:
>> python example.py

In this example, we show how to use disentanglement_lib to:
1. Train a standard VAE (already implemented in disentanglement_lib).
2. Train a custom VAE model.
3. Extract the mean representations for both of these models.
4. Compute the Mutual Information Gap (already implemented) for both models.
5. Compute a custom disentanglement metric for both models.
6. Aggregate the results.
7. Print out the final Pandas data frame with the results.
"""

# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, argparse, yaml
# from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.methods.unsupervised import train
from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
from disentanglement_lib.evaluation.metrics import factor_vae
import tensorflow.compat.v1 as tf
import time
import pdb
import gin.tf
import numpy as np
import torch
from torchvision import models, transforms
import sys 
import random 
# from absl import logging
# from six.moves import range
from torchvision.models import resnet50
import torch.nn as nn
from disentanglement_lib.evaluation.metrics import mig, beta_vae, factor_vae, sap_score


### This part is for ICE-BeeM ######
from models.nets import ConvMLP, FullMLP, SimpleLinear, SimpleEncoder
def feature_net(config):
    if config.model.architecture.lower() == 'convmlp':
        return ConvMLP(config)
    elif config.model.architecture.lower() == 'mlp':
        return FullMLP(config)
    elif config.model.architecture.lower() == 'unet':
        return RefineNetDilated(config)
    elif config.model.architecture == "simple":
        # follow the default encoder we use for other datasets 
        return SimpleEncoder(config)


# @gin.configurable(
#     "evaluation_new", blacklist=["model_dir", "output_dir", "overwrite"])
def evaluate_disentanglement(model_dir,
             output_dir,
             overwrite=False,
             evaluation_fn=gin.REQUIRED,
             random_seed=gin.REQUIRED,
             name="",
             image_size=64,
             gamma=0.1,
             batch_size=64,
             z_dim=50,
             tag="",
             config=None
             ):
    """Loads a representation TFHub module and computes disentanglement metrics.

    Args:
      model_dir: String with path to directory where the representation function
        is saved.
      output_dir: String with the path where the results should be saved.
      overwrite: Boolean indicating whether to overwrite output directory.
      evaluation_fn: Function used to evaluate the representation (see metrics/
        for examples).
      random_seed: Integer with random seed used for training.
      name: Optional string with name of the metric (can be used to name metrics).
    """
    # Delete the output directory if it already exists.
    if tf.gfile.IsDirectory(output_dir):
      if overwrite:
        tf.gfile.DeleteRecursively(output_dir)
      else:
        raise ValueError("Directory {} already exists and overwrite is False.".format(output_dir))
    # Set up time to keep track of elapsed time in results.
    experiment_timer = time.time()

    model_all = feature_net(config)

    def model_representation(data):
        data = torch.from_numpy(data)
        if data.shape[1:] == (64, 64, 1):
          data = torch.permute(data, (0,3,1,2))
          data = data.repeat([1, 3, 1, 1])
        elif data.shape[1:] == (64, 64, 3):
          data = torch.permute(data, (0,3,1,2))
        if torch.cuda.is_available():
          data = data.cuda()
        representation = model(data).squeeze()
        # import pdb; pdb.set_trace()
        return representation.detach().cpu().numpy()
    
    dataset_name = config.data.dataset
    
    epoch_to_eval = [20, 30, 40]
    for epoch in epoch_to_eval:
        print("----- epoch = {} ------".format(epoch))
        checkpoint_path = "run/checkpoints/{}/{}/checkpoint_{}.pth".format(dataset_name, tag, epoch)
        state_dict = torch.load(checkpoint_path)
        weights, optimizer = state_dict
        model_all.load_state_dict(weights)
        model = model_all.encode
        # the line comes from disentanglement_lib/evaluation/metrics/factor_vae_test.py
        # representation_function = lambda x: np.hstack((x,x/2,x))
        if config.metric == "factor_vae":
          scores = factor_vae.compute_factor_vae(dataset, model_representation, 
              np.random.RandomState(0), None, batch_size, 10000, 5000, 10000)
        elif config.metric == "beta_vae":
          scores = beta_vae.compute_beta_vae_sklearn(dataset, model_representation, 
              np.random.RandomState(0), None, batch_size, 10000, 5000)
        elif config.metric == "mig":
          # may still buggy, raises UserWarning: Clustering metrics expects discrete val \
          # ues but received multiclass values for label, and continuous values for target
          def _identity_discretizer(target, num_bins):
            del num_bins
            return target
          gin.bind_parameter("discretizer.discretizer_fn", _identity_discretizer)
          gin.bind_parameter("discretizer.num_bins", 10)
          scores = mig.compute_mig(dataset, model_representation, np.random.RandomState(0), None, 3000)
        elif config.metric == "sap":
          scores = sap_score.compute_sap(dataset, model_representation, np.random.RandomState(0), None, 3000,
            3000, continuous_factors=True)
        elif config.metric == "dci":
          scores = dci.compute_dci(dataset, model_representation, np.random.RandomState(0), None, 1000,
            1000)
          scoers = scores["disentanglement"]
        else:
          NotImplementedError("metric not supported yet")
        print(scores)


def _has_kwarg_or_kwargs(f, kwarg):
    """Checks if the function has the provided kwarg or **kwargs."""
    # For gin wrapped functions, we need to consider the wrapped function.
    if hasattr(f, "__wrapped__"):
      f = f.__wrapped__
    (args, _, kwargs, _, _, _, _) = inspect.getfullargspec(f)
    if kwarg in args or kwargs is not None:
      return True
    return False


# def evaluate_with_gin_new(model_dir,
#                       output_dir,
#                       overwrite=False,
#                       gin_config_files=None,
#                       gin_bindings=None,
#                       image_size=64,
#                       batch_size=64,
#                       gamma=0.1,
#                       z_dim=50,
#                       tag="",
#                       config=None):
#   """Evaluate a representation based on the provided gin configuration.

#   This function will set the provided gin bindings, call the evaluate()
#   function and clear the gin config. Please see the evaluate() for required
#   gin bindings.

#   Args:
#     model_dir: String with path to directory where the representation is saved.
#     output_dir: String with the path where the evaluation should be saved.
#     overwrite: Boolean indicating whether to overwrite output directory.
#     gin_config_files: List of gin config files to load.
#     gin_bindings: List of gin bindings to use.
#   """
#   if gin_config_files is None:
#     gin_config_files = []
#   if gin_bindings is None:
#     gin_bindings = []
#   # gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
#   evaluate_new(model_dir, output_dir, overwrite,image_size=image_size,
#             batch_size=batch_size,gamma=gamma,z_dim=z_dim,tag=tag,config=config)


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='mnist.yaml', help='Path to the config file')
    parser.add_argument('--metric', type=str, required=True)
    args = parser.parse_args()
    return args


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


if __name__ == "__main__":
    args = parse()
    with open(os.path.join('configs', args.config), 'r') as f:
        print('loading config file: {}'.format(os.path.join('configs', args.config)))
        config_raw = yaml.load(f, Loader=yaml.FullLoader)
        config_raw["metric"] = args.metric
    config = dict2namespace(config_raw) 

    dataset_name = config.data.dataset

    # args, config_name = load_yaml_config(command_args.config)
    #args["EVAL"]["ckpoint"] = command_args.ckpoint
    
    seed_torch(0)

    # dataset_name = args['DATALOADER']["dataset"]
    if dataset_name == "3dshapes":
        from datasets.shapes3d import Shapes3D
        dataset = Shapes3D()
    elif dataset_name == "coco":
        from datasets.coco import NewCOCO
        dataset = NewCOCO()
    elif dataset_name == "celeba":
        from datasets.celeba import NewCelebA
        exp = {"dataset":"celeba", "data_path": "data", "img_size": args["MODEL"]["image_size"], "batch_size": args["DATALOADER"]["batch_size"]}
        dataset = NewCelebA(exp, mode="test")
    elif dataset_name.lower() == "dsprites":
        from disentanglement_lib.data.ground_truth import dsprites
        dataset = dsprites.DSprites([1, 2, 3, 4, 5])
    else:
        NotImplementedError("dataset not supported yet")

    gin_bindings = [
      "evaluation.evaluation_fn = @mig",
      "dataset.name='auto'",
      "evaluation.random_seed = 0",
      "mig.num_train=1000",
      "discretizer.discretizer_fn = @histogram_discretizer",
      "discretizer.num_bins = 20"
  ]

    path = "disentanglement_eval/{}/".format(dataset_name)
    result_path = os.path.join(path, "metrics", "MIG")
    representation_path = os.path.join(path, "representation")
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(representation_path, exist_ok=True)
    overwrite=True
    evaluate_disentanglement(model_dir=representation_path,
                            output_dir=result_path,
                            overwrite=overwrite,
                            # gin_bindings=gin_bindings,
                            image_size=config.data.image_size,
                            batch_size=64,
                            gamma=None,
                            z_dim=None,
                            tag="transfer90simple",
                            config=config)
