import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import FastICA
from torch.distributions import Uniform, TransformedDistribution, SigmoidTransform

from models.fce import ebmFCEsegments
from models.nets import MLP_general
from models.nflib.flows import NormalizingFlowModel, Invertible1x1Conv, ActNorm
from models.nflib.spline_flows import NSF_AR

torch.set_default_tensor_type('torch.cuda.FloatTensor')

CKPT_FOLDER = 'run/checkpoints/ivae/'
os.makedirs(CKPT_FOLDER, exist_ok=True)


def ICEBEEM_wrapper(X, Y, ebm_hidden_size, n_layers_ebm, n_layers_flow, lr_flow, lr_ebm,
                    ckpt_file='icebeem.pt', test=False):
    data_dim = X.shape[1]

    model_ebm = MLP_general(input_size=data_dim, hidden_size=[ebm_hidden_size] * n_layers_ebm,
                            n_layers=n_layers_ebm, output_size=data_dim, use_bn=True,
                            activation_function=F.leaky_relu)

    prior = TransformedDistribution(Uniform(torch.zeros(data_dim), torch.ones(data_dim)),
                                    SigmoidTransform().inv)
    nfs_flow = NSF_AR
    flows = [nfs_flow(dim=data_dim, K=8, B=3, hidden_dim=16) for _ in range(n_layers_flow)]
    convs = [Invertible1x1Conv(dim=data_dim) for _ in flows]
    norms = [ActNorm(dim=data_dim) for _ in flows]
    flows = list(itertools.chain(*zip(norms, convs, flows)))
    # construct the model
    model_flow = NormalizingFlowModel(prior, flows)

    pretrain_flow = True
    augment_ebm = True

    # instantiate ebmFCE object
    fce_ = ebmFCEsegments(data=X.astype(np.float32), segments=Y.astype(np.float32),
                          energy_MLP=model_ebm, flow_model=model_flow, verbose=False)

    if pretrain_flow:
        # print('pretraining flow model..')
        fce_.pretrain_flow_model(epochs=1, lr=1e-4)
        # print('pretraining done.')

    # first we pretrain the final layer of EBM model (this is g(y) as it depends on segments)
    fce_.train_ebm_fce(epochs=15, augment=augment_ebm, finalLayerOnly=True, cutoff=.5)

    # then train full EBM via NCE with flow contrastive noise:
    fce_.train_ebm_fce(epochs=50, augment=augment_ebm, cutoff=.5, useVAT=False)

    # evaluate recovery of latents
    recov = fce_.unmixSamples(X, modelChoice='ebm')
    source_est_ica = FastICA().fit_transform((recov))
    recov_sources = [source_est_ica]

    # iterate between updating noise and tuning the EBM
    eps = .025
    for iter_ in range(3):
        # update flow model:
        fce_.train_flow_fce(epochs=5, objConstant=-1., cutoff=.5 - eps, lr=lr_flow)
        # update energy based model:
        fce_.train_ebm_fce(epochs=50, augment=augment_ebm, cutoff=.5 + eps, lr=lr_ebm, useVAT=False)

        # evaluate recovery of latents
        recov = fce_.unmixSamples(X, modelChoice='ebm')
        source_est_ica = FastICA().fit_transform((recov))
        recov_sources.append(source_est_ica)

    return recov_sources