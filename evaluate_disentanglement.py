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
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
from six.moves import range
import tensorflow.compat.v1 as tf
from torchvision.models import resnet50
import torch.nn as nn
from disentanglement_lib.evaluation.metrics import mig, beta_vae, factor_vae, sap_score, dci
import h5py


# issue https://github.com/google-research/disentanglement_lib/issues/18 for the reference 
# of customizing the 3dshapes dataset definition
SHAPES3D_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "3dshapes", "3dshapes.h5"
)


class Shapes3D(ground_truth_data.GroundTruthData):
  """Shapes3D dataset.
  The data set was originally introduced in "Disentangling by Factorising".
  The ground-truth factors of variation are:
  0 - floor color (10 different values)
  1 - wall color (10 different values)
  2 - object color (10 different values)
  3 - object size (8 different values)
  4 - object type (4 different values)
  5 - azimuth (15 different values)
  """

  def __init__(self):
    # with tf.gfile.GFile(SHAPES3D_PATH, "rb") as f:
    #   # Data was saved originally using python2, so we need to set the encoding.
    #   data = np.load(f, encoding="latin1")
    # images = data["images"]
    # labels = data["labels"]
    # n_samples = np.prod(images.shape[0:6])
    with h5py.File(SHAPES3D_PATH, 'r') as dataset:
      images = dataset['images'][()]
      labels = dataset['labels'][()]
      n_samples = images.shape[0]

    self.images = (
        images.reshape([n_samples, 64, 64, 3]).astype(np.float32) / 255.)
    features = labels.reshape([n_samples, 6])
    self.factor_sizes = [10, 10, 10, 8, 4, 15]
    self.latent_factor_indices = list(range(6))
    self.num_total_factors = features.shape[1]
    self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                    self.latent_factor_indices)
    self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
        self.factor_sizes)

  @property
  def num_factors(self):
    return self.state_space.num_latent_factors

  @property
  def factors_num_values(self):
    return self.factor_sizes

  @property
  def observation_shape(self):
    return [64, 64, 3]


  def sample_factors(self, num, random_state):
    """Sample a batch of factors Y."""
    return self.state_space.sample_latent_factors(num, random_state)

  def sample_observations_from_factors(self, factors, random_state):
    all_factors = self.state_space.sample_all_factors(factors, random_state)
    indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
    return self.images[indices]



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
             config=None,
             dataset=None
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
          # data = data.repeat([1, 3, 1, 1])
        elif data.shape[1:] == (64, 64, 3):
          data = torch.permute(data, (0,3,1,2))
        if torch.cuda.is_available():
          data = data.cuda()
        representation = model(data).squeeze()
        # import pdb; pdb.set_trace()
        return representation.detach().cpu().numpy()
    
    dataset_name = config.data.dataset
    
    if dataset_name.lower() == "dsprites":
      epoch_to_eval = [30]
    elif dataset_name.lower() == "smallnorb":
      epoch_to_eval = [20, 40, 60, 80, 100, 120, 140, 160]
    elif dataset_name.lower() == "cars3d":
      epoch_to_eval = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900]
    elif dataset_name.lower() == "3dshapes":
      epoch_to_eval = [100]

    for epoch in epoch_to_eval:
        print("----- epoch = {} | metric = {} | dataset = {} | trial = {} ------".format(epoch, config.metric, dataset_name,  config.trial_name))
        checkpoint_path = "run/checkpoints/{}/{}/{}/checkpoint_{}.pth".format(dataset_name, tag, config.trial_name, epoch)
        print("Load checkpoint from ", checkpoint_path)
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

          def _histogram_discretize(target, num_bins):
            """Discretization based on histograms."""
            discretized = np.zeros_like(target)
            for i in range(target.shape[0]):
              discretized[i, :] = np.digitize(target[i, :], np.histogram(
                  target[i, :], num_bins)[1][:-1])
            return discretized
          # gin_bindings = [
          #     # "evaluation.evaluation_fn = @mig",
          #     # "evaluation.random_seed = 0",
          #     "mig.num_train=10000",
          #     "discretizer.discretizer_fn = @histogram_discretizer",
          #     "discretizer.num_bins = 20"
          # ]
          # if gin_config_files is None:
          # gin_config_files = []
          # if gin_bindings is None:
          #   gin_bindings = []
          # gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
          gin.bind_parameter("discretizer.discretizer_fn",  _histogram_discretize)
          gin.bind_parameter("discretizer.num_bins", 20)
          # gin.bind_parameter(gin_bindings)
          scores = mig.compute_mig(dataset, model_representation, np.random.RandomState(0), None, num_train=10000, batch_size=16)
        elif config.metric == "sap":
          scores = sap_score.compute_sap(dataset, model_representation, np.random.RandomState(0), None, num_train=10000,
            num_test=16, continuous_factors=True)
        elif config.metric == "dci":
          scores = dci.compute_dci(dataset, model_representation, np.random.RandomState(0), None, num_train=10000, num_test=5000)
          scores = scores["disentanglement"]
        else:
          NotImplementedError("metric not supported yet")
        print(scores)
        with open("{}_eval_log.txt".format(dataset_name), 'a') as f:
          f.write("------------- Trial {} -----------\n".format(config.trial_name))
          f.write("------------- {} --------------\n".format(config.metric))
          if config.metric == "dci":
            f.write("{}\n".format(scores))
          else:
            for k, v in scores.items():
              f.write("{}: {}\n".format(k, v))
          f.flush()


def _has_kwarg_or_kwargs(f, kwarg):
    """Checks if the function has the provided kwarg or **kwargs."""
    # For gin wrapped functions, we need to consider the wrapped function.
    if hasattr(f, "__wrapped__"):
      f = f.__wrapped__
    (args, _, kwargs, _, _, _, _) = inspect.getfullargspec(f)
    if kwarg in args or kwargs is not None:
      return True
    return False


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
    parser.add_argument('--metric', type=str)
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


def main(args):
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
    if dataset_name.lower() == "3dshapes":
        # from datasets.shapes3d import Shapes3D
        # dataset = Shapes3D()
        # from disentanglement_lib.data.ground_truth import shapes3d
        dataset = Shapes3D()
    elif dataset_name.lower() == "smallnorb":
        SMALLNORB_TEMPLATE = os.path.join(
            os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "smallnorb", "raw"
            "smallnorb-{}-{}.mat")

        SMALLNORB_CHUNKS = [
            "5x46789x9x18x6x2x96x96-training",
            "5x01235x9x18x6x2x96x96-testing",
        ]
        from disentanglement_lib.data.ground_truth import norb
        dataset = norb.SmallNORB() 
    elif dataset_name.lower() == "cars3d":
        # from disentanglement_lib.data.ground_truth import cars3d
        from data.cars3d import Cars3D_tensorflow
        dataset = Cars3D_tensorflow(root="datasets", img_size=64)
    # elif dataset_name == "coco":
    #     from datasets.coco import NewCOCO
    #     dataset = NewCOCO()
    # elif dataset_name == "celeba":
    #     from datasets.celeba import NewCelebA
    #     exp = {"dataset":"celeba", "data_path": "data", "img_size": args["MODEL"]["image_size"], "batch_size": args["DATALOADER"]["batch_size"]}
    #     dataset = NewCelebA(exp, mode="test")
    elif dataset_name.lower() == "dsprites":
        from disentanglement_lib.data.ground_truth import dsprites
        dataset = dsprites.DSprites([1, 2, 3, 4, 5])
    else:
        NotImplementedError("dataset not supported yet")

    # For the best practice of DisLib we should use gin to pass configurations from begining to end,
    # but for quicker adaptaton, I disable this temporally
    # gin_bindings = [
    #   "evaluation.evaluation_fn = @mig",
    #   "dataset.name='auto'",
    #   "evaluation.random_seed = 0",
    #   "mig.num_train=1000",
    #   "discretizer.discretizer_fn = @histogram_discretizer",
    #   "discretizer.num_bins = 20"
    # ]

    path = "disentanglement_eval/{}/".format(dataset_name)
    result_path = os.path.join(path, "metrics", "MIG")
    representation_path = os.path.join(path, "representation")
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(representation_path, exist_ok=True)
    overwrite=True

    for trial in ["dim10_1", "dim10_4"]:
      config.trial_name = trial
      evaluate_disentanglement(model_dir=representation_path,
                              output_dir=result_path,
                              overwrite=overwrite,
                              # gin_bindings=gin_bindings, # we should bring this back 
                              image_size=config.data.image_size,
                              batch_size=64,
                              gamma=None,
                              z_dim=None,
                              tag="transfera90simple",
                              config=config,
                              dataset=dataset)


if __name__ == "__main__":
  args = parse()
  # for config_name in ["dsprites.yaml", "smallnorb.yaml", "shapes3d.yaml"]:
  for config_name in ["smallnorb.yaml"]:
    for metric_name in ["beta_vae", "factor_vae", "mig", "sap", "dci"]:
      args.config = config_name 
      args.metric = metric_name
      main(args)