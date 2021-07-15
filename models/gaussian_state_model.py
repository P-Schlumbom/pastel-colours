import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import pickle

VERBOSE = 1

class PosteriorTransitionMachine(nn.Module):
    def __init__(self, b_prior, c_prior, d_prior, n_states=3):
        super(PosteriorTransitionMachine, self).__init__()

        self.ab = nn.Linear(n_states, n_states)
        self.ac = nn.Linear(n_states, n_states)
        self.ad = nn.Linear(n_states, n_states)
        self.bc = nn.Linear(n_states, n_states)
        self.bd = nn.Linear(n_states, n_states)
        self.cd = nn.Linear(n_states, n_states)

        self.b_prior = torch.from_numpy(b_prior)
        self.c_prior = torch.from_numpy(c_prior)
        self.d_prior = torch.from_numpy(d_prior)

    def forward(self, a_posteriors, b_priors, c_priors, d_priors):
        b_posteriors = self.ab(a_posteriors) * b_priors
        b_posteriors = b_posteriors / torch.sum(b_posteriors)
        c_posteriors = self.ac(a_posteriors) * self.bc(b_posteriors) * c_priors
        c_posteriors = c_posteriors / torch.sum(c_posteriors)
        d_posteriors = self.ad(a_posteriors) * self.bd(b_posteriors) * self.cd(c_posteriors) * d_priors
        d_posteriors = d_posteriors / torch.sum(d_posteriors)
        posterior_predictions = torch.cat((a_posteriors, b_posteriors, c_posteriors, d_posteriors), dim=1)
        print(posterior_predictions.size())
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
        n_transitions = M - 1
        for transitions in range(M - 1, 0, -1):
            n_transitions += transitions

        if isinstance(C, int):
            self.gmms = [GaussianMixture(n_components=C) for m in range(M)]
            self.transition_matrices = {m : [np.zeros((C, C)) for n in range(M-m-1, 0, -1)] for m in range(M)}
            self.transition_biases = {m: [np.zeros(C) for n in range(M-m-1, 0, -1)] for m in range(M)}
        elif isinstance(C, dict):
            self.gmms = [GaussianMixture(n_components=C[m]) for m in range(M)]
            self.transition_matrices = [np.zeros((C, C)) for n in n_transitions]  # not completed
        else:
            raise Exception("C must be integer of list of integers!")

    def save_model(self, savepath="test.pkl"):
        pickle.dump(self.gmms, open(savepath, 'wb'))

    def load_model(self, loadpath="test.pkl"):
        self.gmms = pickle.load(open(loadpath, 'rb'))

    def train_state(self, state, X):
        self.gmms[state].fit(X)

    def train_GD(self, P, priors, epochs=10):
        """
        Train transition matrices on dataset of posteriors with gradient descent
        :param epochs:
        :param P: list of M NxC numpy arrays, one posterior for each sample for each state
        :return:
        """
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True

        learning_rate = 0.0001
        epochs = epochs

        model = PosteriorTransitionMachine(self.gmms[1].weights_, self.gmms[2].means_, self.gmms[3].means_, n_states=self.C)
        model = model.to(device)
        model_optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

        loss = nn.MSELoss()

        # prepare posteriors and priors
        P_mat = np.concatenate(P, axis=1)
        print(P_mat.shape)

        # training
        # for now, train on whole dataset at once (batch_size=N)
        N = P[0].shape[0]
        a_pos, b_prior, c_prior, d_prior = P[0], priors[1], priors[2], priors[3]
        a_pos, b_prior, c_prior, d_prior = torch.from_numpy(a_pos), torch.from_numpy(b_prior), \
                                           torch.from_numpy(c_prior), torch.from_numpy(d_prior)
        P_mat = torch.from_numpy(P_mat)
        a_pos, b_prior, c_prior, d_prior, P_mat = a_pos.to(device, dtype=torch.float), \
                                                  b_prior.to(device, dtype=torch.float), \
                                                  c_prior.to(device, dtype=torch.float), \
                                                  d_prior.to(device, dtype=torch.float), \
                                                  P_mat.to(device, dtype=torch.float)
        for epoch in range(epochs):
            total_loss = 0
            model_optimiser.zero_grad()
            pos_pred = model(a_pos, b_prior, c_prior, d_prior)
            pos_loss = loss(pos_pred, P_mat)
            pos_loss.backward()
            model_optimiser.step()

            total_loss = pos_loss.item()
            epoch_loss = total_loss / N
            if VERBOSE >= 1: print("epoch {}: model loss = {:.5g}".format(epoch, epoch_loss))

        # assign weight matrices
        self.transition_matrices[0][0] = model.ab.weight.cpu().detach().numpy()
        self.transition_matrices[0][1] = model.ac.weight.cpu().detach().numpy()
        self.transition_matrices[0][2] = model.ad.weight.cpu().detach().numpy()
        self.transition_matrices[1][0] = model.bc.weight.cpu().detach().numpy()
        self.transition_matrices[1][1] = model.bd.weight.cpu().detach().numpy()
        self.transition_matrices[2][0] = model.cd.weight.cpu().detach().numpy()
        self.transition_biases[0][0] = model.ab.bias.cpu().detach().numpy()
        self.transition_biases[0][1] = model.ac.bias.cpu().detach().numpy()
        self.transition_biases[0][2] = model.ad.bias.cpu().detach().numpy()
        self.transition_biases[1][0] = model.bc.bias.cpu().detach().numpy()
        self.transition_biases[1][1] = model.bd.bias.cpu().detach().numpy()
        self.transition_biases[2][0] = model.cd.bias.cpu().detach().numpy()

    def train_model(self, X):
        """
        Build all Gaussian Mixture Models at once.
        :param X: list of numpy arrays, where the array at index i is the data for state i in the shape (N, D),
        containing N D-dimensional data points.
        :return:
        """
        N, D = X[0].shape
        # 1. prepare prior GMM models
        for m in range(self.M):
            self.train_state(m, X[m])

        # 2. compute dataset of posteriors
        P = []
        priors = []
        for m in range(self.M):
            P.append(self.gmms[m].predict_proba(X[m]))
            print(np.expand_dims(self.gmms[m].weights_, axis=0).shape)
            priors.append(np.repeat(np.expand_dims(self.gmms[m].weights_, axis=0), repeats=N, axis=0))

        # 3. train transition matrices with gradient descent
        self.train_GD(P, priors)

    def sample_from_multigauss(self, id, weights):
        print(weights)
        target_gauss = np.random.choice(np.arange(self.C), p=weights)  # select the gaussian for the current GMM
        # to sample from
        means = self.gmms[id].means_[target_gauss]
        covars = self.gmms[id].covariances_[target_gauss]
        sample = np.random.multivariate_normal(means, covars)
        return sample

    def sample(self, n_samples=1, set_values={}):
        """
        Sample a value from each state
        :param n_samples:
        :param set_values: optional, a dictionary of numpy arrays of filled values indexed by the corresponding state.
        :return:
        """
        #TODO: complete this. Basically, for each state, sample a gaussian from the Gaussian mixture in proportion to
        # its weight, and then sample a value from it according to the Gaussian's parameters.
        # this might help: https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html

        # compute posteriors
        posteriors = np.ones((self.M, self.C))
        for m in range(self.M):
            posteriors[m, :] = self.gmms[m].weights_.copy()  # assign priors first
        print(posteriors)

        output = np.ones((self.M, 3)) * -1  # to be filled
        for m in range(self.M):
            if m in set_values.keys():
                output[m, :] = set_values[m]
                posteriors[m, :] = self.gmms[m].predict_proba(set_values[m])
            else:
                print(posteriors, np.sum(posteriors, axis=1))
                #posteriors[m, :] /= np.sum(posteriors[m, :])  # normalise
                posteriors[m, :] = (posteriors[m, :] - np.min(posteriors[m, :])) / (np.max(posteriors[m, :]) - np.min(posteriors[m, :]))
                posteriors[m, :] /= np.sum(posteriors[m, :])  # normalise
                print(posteriors, np.sum(posteriors, axis=1))
                output[m, :] = self.sample_from_multigauss(m, posteriors[m, :])
                print("---")
                for n in range(len(self.transition_matrices[m])):
                    #print(posteriors[m, :], self.transition_matrices[m][n])
                    posteriors[m+n+1] *= (np.matmul(posteriors[m, :], self.transition_matrices[m][n].T) + self.transition_biases[m][n])
                    #posteriors[m + n + 1] *= np.matmul(self.transition_matrices[m][n], posteriors[m, :])
                    #print(posteriors[m+n+1])
                    #posteriors[m + n + 1] /= np.sum(posteriors[m+n+1, :])
                    #print(posteriors[m+n+1])

        return output

