import numpy as np
from sklearn.mixture import GaussianMixture
import pickle

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