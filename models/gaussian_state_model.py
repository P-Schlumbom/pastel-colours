import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import pickle

class PosteriorTransitionMachine(nn.Module):
    def __init__(self, n_states=3):
        super(PosteriorTransitionMachine, self).__init__()

        self.ab = nn.Linear(n_states, n_states)
        self.ac = nn.Linear(n_states, n_states)
        self.ad = nn.Linear(n_states, n_states)
        self.bc = nn.Linear(n_states, n_states)
        self.bd = nn.Linear(n_states, n_states)
        self.cd = nn.Linear(n_states, n_states)

    def forward(self, a_posteriors, b_priors, c_priors, d_priors):
        b_posteriors = self.ab(a_posteriors) * b_priors
        b_posteriors = b_posteriors / torch.sum(b_posteriors)
        c_posteriors = self.ac(a_posteriors) * self.bc(b_posteriors) * c_priors
        c_posteriors = c_posteriors / torch.sum(c_posteriors)
        d_posteriors = self.ad(a_posteriors) * self.bd(b_posteriors) * self.cd(c_posteriors) * d_priors
        d_posteriors = d_posteriors / torch.sum(d_posteriors)
        posterior_predictions = torch.cat((a_posteriors, b_posteriors, c_posteriors, d_posteriors))
        return posterior_predictions

class GaussianStateMachine():
    def __init__(self, M, C):
        """
        Initialise the Gaussian State Machine. In this machine, each state has an incoming connection from every
        previous state (i.e. all states with a lower index). It is probably not well suited for longer state chains,
        which might be better off making the Markov assumption.
        :param M: the number of states
        :param C: int or list, the number of components per state. If a list, the values are the number of components
        for the state indexed by that position.
        """
        self.M = M
        self.C = C

        if isinstance(C, int):
            self.gmms = [GaussianMixture(n_components=C) for m in range(M)]
        elif isinstance(C, dict):
            self.gmms = [GaussianMixture(n_components=C[m]) for m in range(M)]
        else:
            raise Exception("C must be integer of list of integers!")

    def save_model(self, savepath="test.pkl"):
        pickle.dump(self.gmms, open(savepath, 'wb'))

    def load_model(self, loadpath="test.pkl"):
        self.gmms = pickle.load(open(loadpath, 'rb'))

    def train_state(self, state, X):
        self.gmms[state].fit(X)

    def train_model(self, X):
        """
        Build all Gaussian Mixture Models at once.
        :param X: list of numpy arrays, where the array at index i is the data for state i in the shape (N, D),
        containing N D-dimensional data points.
        :return:
        """
        for m in range(self.M):
            self.train_state(m, X[m])

    def sample(self, n_samples=1, set_states=None):
        """
        Sample a value from each state
        :param n_samples:
        :param set_states: optional, a numpy array with some values already entered. Unfilled values should be
        represented by -1.
        :return:
        """
        #TODO: complete this. Basically, for each state, sample a gaussian from the Gaussian mixture in proportion to
        # its weight, and then sample a value from it according to the Gaussian's parameters.
        # this might help: https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html
        pass