from __future__ import absolute_import, division, print_function

import numpy as np
import random
import sys
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
#%matplotlib inline

import logging
import torch
from torch import nn
import scipy
import re
import os
import joblib

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

device = 'cpu'
#print("test")
torch.manual_seed(1)

# Global variables

ACTIVATION_DICT = {"ReLU": torch.nn.ReLU(),
                   "Hardtanh": torch.nn.Hardtanh(),
                   "ReLU6": torch.nn.ReLU6(),
                   "Sigmoid": torch.nn.Sigmoid(),
                   "Tanh": torch.nn.Tanh(),
                   "ELU": torch.nn.ELU(),
                   "CELU": torch.nn.CELU(),
                   "SELU": torch.nn.SELU(),
                   "GLU": torch.nn.GLU(),
                   "LeakyReLU": torch.nn.LeakyReLU(),
                   "LogSigmoid": torch.nn.LogSigmoid(),
                   "Softplus": torch.nn.Softplus()}


def build_network(network_name, params):

    if network_name=="feedforward":

        net = feedforward_network(params)

    return net


def feedforward_network(params):

    """Architecture for a Feedforward Neural Network

    Args:

        ::params::

        ::params["input_dim"]::
        ::params[""rep_dim""]::
        ::params["num_hidden"]::
        ::params["activation"]::
        ::params["num_layers"]::
        ::params["dropout_prob"]::
        ::params["dropout_active"]::
        ::params["LossFn"]::

    Returns:

        ::_architecture::

    """

    modules          = []

    if params["dropout_active"]:

        modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

    # Input layer

    modules.append(torch.nn.Linear(params["input_dim"], params["num_hidden"],bias=False))
    modules.append(ACTIVATION_DICT[params["activation"]])

    # Intermediate layers

    for u in range(params["num_layers"] - 1):

        if params["dropout_active"]:

            modules.append(torch.nn.Dropout(p=params["dropout_prob"]))

        modules.append(torch.nn.Linear(params["num_hidden"], params["num_hidden"],
                                       bias=False))
        modules.append(ACTIVATION_DICT[params["activation"]])


    # Output layer

    modules.append(torch.nn.Linear(params["num_hidden"], params["rep_dim"],bias=False))

    _architecture    = nn.Sequential(*modules)

    return _architecture

"""

  -----------------------------------------
  One-class representations
  -----------------------------------------

"""

from torch.autograd import Variable

# One-class loss functions
# ------------------------


def OneClassLoss(outputs, c):

    dist   = torch.sum((outputs - c) ** 2, dim=1)
    loss   = torch.mean(dist)

    return loss


def SoftBoundaryLoss(outputs, R, c, nu):

    dist   = torch.sum((outputs - c) ** 2, dim=1)
    scores = dist - R ** 2
    loss   = R ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))

    scores = dist
    loss   = (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))

    return loss


LossFns    = dict({"OneClass": OneClassLoss, "SoftBoundary": SoftBoundaryLoss})

# Base network
# ---------------------

class BaseNet(nn.Module):

    """Base class for all neural networks."""

    def __init__(self):

        super().__init__()

        self.logger  = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the last layer

    def forward(self, *input):

        """Forward pass logic

        :return: Network output
        """
        raise NotImplementedError

    def summary(self):

        """Network summary."""

        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params         = sum([np.prod(p.size()) for p in net_parameters])

        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


def get_radius(dist:torch.Tensor, nu:float):

    """Optimally solve for radius R via the (1-nu)-quantile of distances."""

    return np.quantile(np.sqrt(dist.clone().data.float().numpy()), 1 - nu)

class OneClassLayer(BaseNet):

    def __init__(self, params=None, hyperparams=None):

        super().__init__()

        # set all representation parameters - remove these lines

        self.rep_dim        = params["rep_dim"]
        self.input_dim      = params["input_dim"]
        self.num_layers     = params["num_layers"]
        self.num_hidden     = params["num_hidden"]
        self.activation     = params["activation"]
        self.dropout_prob   = params["dropout_prob"]
        self.dropout_active = params["dropout_active"]
        self.loss_type      = params["LossFn"]
        self.train_prop     = params['train_prop']
        self.learningRate   = params['lr']
        self.epochs         = params['epochs']
        self.warm_up_epochs = params['warm_up_epochs']
        self.weight_decay   = params['weight_decay']
        if torch.cuda.is_available():
            self.device     = torch.device('cuda') # Make this an option
        else:
            self.device     = torch.device('cpu')
        # set up the network

        self.model          = build_network(network_name="feedforward", params=params).to(self.device)

        # create the loss function

        self.c              = hyperparams["center"].to(self.device)
        self.R              = hyperparams["Radius"]
        self.nu             = hyperparams["nu"]

        self.loss_fn        = LossFns[self.loss_type]


    def forward(self, x):

        x                   = self.model(x)

        return x


    def fit(self, x_train, verbosity=True):


        self.optimizer      = torch.optim.AdamW(self.model.parameters(), lr=self.learningRate, weight_decay = self.weight_decay)
        self.X              = torch.tensor(x_train.reshape((-1, self.input_dim))).float()

        if self.train_prop != 1:
            x_train, x_val = x_train[:int(self.train_prop*len(x_train))], x_train[int(self.train_prop*len(x_train)):]
            inputs_val = Variable(torch.from_numpy(x_val).to(self.device)).float()

        self.losses         = []
        self.loss_vals       = []


        for epoch in range(self.epochs):

            # Converting inputs and labels to Variable

            inputs = Variable(torch.from_numpy(x_train)).to(self.device).float()

            self.model.zero_grad()

            self.optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = self.model(inputs)

            # get loss for the predicted output

            if self.loss_type=="SoftBoundary":

                self.loss = self.loss_fn(outputs=outputs, R=self.R, c=self.c, nu=self.nu)

            elif self.loss_type=="OneClass":

                self.loss = self.loss_fn(outputs=outputs, c=self.c)


            #self.c    = torch.mean(torch.tensor(outputs).float(), dim=0)

            # get gradients w.r.t to parameters
            self.loss.backward(retain_graph=True)
            self.losses.append(self.loss.detach().cpu().numpy())

            # update parameters
            self.optimizer.step()

            if (epoch >= self.warm_up_epochs) and (self.loss_type=="SoftBoundary"):

                dist   = torch.sum((outputs - self.c) ** 2, dim=1)
                #self.R = torch.tensor(get_radius(dist, self.nu))

            if self.train_prop != 1.0:
                with torch.no_grad():

                    # get output from the model, given the inputs
                    outputs = self.model(inputs_val)

                    # get loss for the predicted output

                    if self.loss_type=="SoftBoundary":

                        loss_val = self.loss_fn(outputs=outputs, R=self.R, c=self.c, nu=self.nu)

                    elif self.loss_type=="OneClass":

                        loss_val = self.loss_fn(outputs=outputs, c=self.c).detach.cpu().numpy()

                    self.loss_vals.append(loss_val)




            if verbosity:
                if self.train_prop == 1:
                    print('epoch {}, loss {}'.format(epoch, self.loss.item()))
                else:
                    print('epoch {:4}, train loss {:.4e}, val loss {:.4e}'.format(epoch, self.loss.item(),loss_val))


def tune_nearest_neighbors(embedded_real_data_train, embedded_real_data_test, radius, alpha):

    k_vals                   = list(range(2, 10))
    epsilon                  = 0.025
    errs                     = []

    for k in k_vals:

        errs.append(evaluate_beta_recall_kNN(embedded_real_data_train, embedded_real_data_test, radius, alpha, k))

    k_opt = k_vals[np.argmin(np.abs(np.array(errs) - (alpha * (1-epsilon))))]

    return k_opt

def evaluate_beta_recall_kNN(embedded_real_data, embedded_synth_data, radius, alpha, k):

    synth_center             = torch.tensor(np.mean(embedded_synth_data, axis=0)).float()

    # fit nearest neighbor for a given k
    #print(np.shape(embedded_real_data))
    nbrs_real                = NearestNeighbors(n_neighbors=k, n_jobs=-1, p=2).fit(embedded_real_data)
    real_to_real, _          = nbrs_real.kneighbors(embedded_real_data)

    # match to synthetic data

    nbrs_synth               = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(embedded_synth_data)
    real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(embedded_real_data)

    # Let us find the closest real point to any real point, excluding itself (therefore 1 instead of 0)
    real_to_real          = torch.from_numpy(real_to_real[:, -1].squeeze())
    real_to_synth         = torch.from_numpy(real_to_synth.squeeze())
    real_to_synth_args    = real_to_synth_args.squeeze()

    real_synth_closest    = embedded_synth_data[real_to_synth_args]

    real_synth_closest_d  = torch.sqrt(torch.sum((torch.tensor(real_synth_closest).float() - synth_center) ** 2, dim=1))

    closest_synth_Radii   = np.quantile(torch.sqrt(torch.sum((torch.tensor(embedded_synth_data).float() - synth_center) ** 2, dim=1)), [alpha])
    beta_coverage         = np.mean(((real_to_synth <= real_to_real) * (real_synth_closest_d <= closest_synth_Radii[0])).detach().float().numpy())

    return beta_coverage

def compute_beta_recall_kNN(real_data, synthetic_data, real_embedding, synthetic_embedding, n_steps):


    alphas         = np.linspace(0, 1, n_steps)

    real_data      = embed_data_numpy(real_data, real_embedding)
    synthetic_data = embed_data_numpy(synthetic_data, real_embedding)
    emb_center     = torch.tensor(real_embedding.c, device=device)

    if synthetic_embedding is not None:

        synth_center   = torch.tensor(synthetic_embedding.c, device=device)

    Radii                 = np.quantile(torch.sqrt(torch.sum((torch.tensor(real_data).float() - emb_center) ** 2, dim=1)), alphas)

    synth_center          = torch.tensor(np.mean(synthetic_data, axis=0)).float()

    alpha_precision_curve = []
    beta_recall_curve     = []

    synth_to_center       = torch.sqrt(torch.sum((torch.tensor(synthetic_data).float() - emb_center) ** 2, dim=1))


    for u in range(len(Radii)):

        precision_audit_mask = (synth_to_center <= Radii[u]).detach().float().numpy()
        alpha_precision      = np.mean(precision_audit_mask)

        n_data               = real_data.shape[0]

        #print(f"n_data{n_data}")

        real_data_train      = real_data[: int(np.floor(n_data/2)), :]
        real_data_test       = real_data[int(np.floor(n_data/2)) + 1:, :]

        # Tune the k-NN hyperparameter
        k_opt                = tune_nearest_neighbors(real_data_train, real_data_test, Radii[u], alphas[u])

        #print(f" real data shape {real_data.shape}")
        beta_recall         = evaluate_beta_recall_kNN(real_data, synthetic_data, Radii[u], alphas[u], k=k_opt)

        #alpha_precision_curve.append(alpha_precision)
        beta_recall_curve.append(beta_recall)


    return alphas,  beta_recall_curve

def compute_beta_recall_OC(real_data, synthetic_data, real_embedding, synthetic_embedding, n_steps):

    alphas          = np.linspace(0, 1, n_steps)

    # find the centers of the hyperspheres

    real_emb_center = torch.tensor(real_embedding.c, device=device)

    if synthetic_embedding is not None:

        synth_emb_center   = torch.tensor(synthetic_embedding.c, device=device)

    # embedd real and synthetic data

    real_data_real_embed   = embed_data_numpy(real_data, real_embedding)
    real_data_synth_embed  = embed_data_numpy(real_data, synthetic_embedding)
    synth_data_real_embed  = embed_data_numpy(synthetic_data, real_embedding)
    synth_data_synth_embed = embed_data_numpy(synthetic_data, synthetic_embedding)

    # evaluate real and synthetic radii within the hyperspheres

    real_Radii  = np.quantile(torch.sqrt(torch.sum((torch.tensor(real_data_real_embed).float() - real_emb_center) ** 2, dim=1)), alphas)
    synth_Radii = np.quantile(torch.sqrt(torch.sum((torch.tensor(synth_data_synth_embed).float() - synth_emb_center) ** 2, dim=1)), alphas)

    synth_to_real_center = torch.sqrt(torch.sum((torch.tensor(synth_data_real_embed).float() - real_emb_center) ** 2, dim=1))
    real_to_synth_center = torch.sqrt(torch.sum((torch.tensor(real_data_synth_embed).float() - synth_emb_center) ** 2, dim=1))

    # evaluate the precision and recall curves
    beta_recall_curve    = []

    for k in range(len(real_Radii)):


        recall_audit_mask    = (real_to_synth_center <= synth_Radii[k]).detach().float().numpy()
        beta_recall          = np.mean(recall_audit_mask)

        beta_recall_curve.append(beta_recall)


    return alphas, beta_recall_curve

def compute_alpha_precision(real_data, synthetic_data, real_embedding, n_steps):

    emb_center = torch.tensor(real_embedding.c, device=device)

    nn_size = 2
    alphas  = np.linspace(0, 1, n_steps)


    Radii   = np.quantile(torch.sqrt(torch.sum((torch.tensor(real_data).float() - emb_center) ** 2, dim=1)), alphas)

    synth_center          = torch.tensor(np.mean(synthetic_data, axis=0)).float()

    alpha_precision_curve = []
    beta_coverage_curve   = []

    synth_to_center       = torch.sqrt(torch.sum((torch.tensor(synthetic_data).float() - emb_center) ** 2, dim=1))


    nbrs_real = NearestNeighbors(n_neighbors = 2, n_jobs=-1, p=2).fit(real_data)
    real_to_real, _       = nbrs_real.kneighbors(real_data)

    nbrs_synth = NearestNeighbors(n_neighbors = 1, n_jobs=-1, p=2).fit(synthetic_data)
    real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(real_data)

    # Let us find the closest real point to any real point, excluding itself (therefore 1 instead of 0)
    real_to_real          = torch.from_numpy(real_to_real[:,1].squeeze())
    real_to_synth         = torch.from_numpy(real_to_synth.squeeze())
    real_to_synth_args    = real_to_synth_args.squeeze()

    real_synth_closest    = synthetic_data[real_to_synth_args]

    real_synth_closest_d  = torch.sqrt(torch.sum((torch.tensor(real_synth_closest).float()- synth_center) ** 2, dim=1))
    closest_synth_Radii   = np.quantile(real_synth_closest_d, alphas)



    for k in range(len(Radii)):
        precision_audit_mask = (synth_to_center <= Radii[k]).detach().float().numpy()
        alpha_precision      = np.mean(precision_audit_mask)

        beta_coverage        = np.mean(((real_to_synth <= real_to_real) * (real_synth_closest_d <= closest_synth_Radii[k])).detach().float().numpy())

        alpha_precision_curve.append(alpha_precision)
        beta_coverage_curve.append(beta_coverage)


    # See which one is bigger
    try:
        authen = real_to_real[real_to_synth_args] < real_to_synth
        authenticity = np.mean(authen.numpy())
    except:
        authenticity = None

    Delta_precision_alpha = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(alpha_precision_curve))) * (alphas[1] - alphas[0])
    Delta_coverage_beta  = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(beta_coverage_curve))) * (alphas[1] - alphas[0])

    return alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, authenticity

# OC parameters

params  = dict({"rep_dim": None,
                "num_layers": 2,
                "num_hidden": 200,
                "activation": "ReLU",
                "dropout_prob": 0.5,
                "dropout_active": False,
                "train_prop" : 1,
                "epochs" : 100,
                "warm_up_epochs" : 10,
                "lr" : 1e-3,
                "weight_decay" : 1e-2,
                "LossFn": "SoftBoundary"})

hyperparams = dict({"Radius": 1, "nu": 1e-2})


def compute_metrics(X, Y, embed_synthetic=False, steps =30, verbosity=False):

    params["input_dim"]   = X.shape[1]
    params["rep_dim"]     = X.shape[1]
    hyperparams["center"] = torch.ones(X.shape[1])

    real_embedding        = OneClassLayer(params=params, hyperparams=hyperparams)
    real_embedding.fit(X, verbosity=verbosity)

    if embed_synthetic:

        hyperparams["center"] = torch.ones(X.shape[1]) #torch.tensor(np.mean(Y, axis=0)) #*
        synthetic_embedding   = OneClassLayer(params=params, hyperparams=hyperparams)
        synthetic_embedding.fit(Y, verbosity=verbosity)

    else:
        synthetic_embedding = None

    if embed_synthetic:

        alphas_, beta_recall_curve = compute_beta_recall_OC(X, Y, real_embedding, synthetic_embedding, n_steps=steps)

    else:

        alphas_, beta_recall_curve = compute_beta_recall_kNN(X, Y, real_embedding, synthetic_embedding,n_steps=steps)
    alphas_a_, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, authen = compute_alpha_precision(X,Y,real_embedding,n_steps=steps)
    #print(beta_coverage_curve == beta_recall_curve)

    '''from dataclasses import dataclass
    @dataclass
    class AlphaBetaAuth:
        alphas :np.ndarray = alphas_
        beta_recall:np.ndarray = beta_recall_curve
        alphas_a:np.ndarray = alphas_a_
        alpha_precision:np.ndarray = alpha_precision_curve
        beta_coverage:np.ndarray = beta_coverage_curve
        delta_alpha:np.ndarray = Delta_precision_alpha
        delta_beta:np.ndarray = Delta_coverage_beta
        authenticity:np.ndarray = authen'''
    alpha_beta_auth = {
        "alphas" : alphas_,
        "beta_recall" : beta_recall_curve,
        "alphas_a" : alphas_a_,
        "alpha_precision" : alpha_precision_curve,
        "beta_coverage" : beta_coverage_curve,
        "delta_alpha" : Delta_precision_alpha,
        "delta_beta" : Delta_coverage_beta,
        "authenticity" : authen
    }

    return alpha_beta_auth


def embed_data_numpy(data, embedding):

    return embedding(torch.tensor(data).float()).float().detach().numpy()