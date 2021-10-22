import torch
import copy
from daphne import daphne
import torch.distributions as dist
from graph_based_sampling import deterministic_eval, sample_vars_from_joint

class HMC:
    def __init__(self, graph):
        self.graph = graph[1]
        self.expr = graph[2]
        self.X = [v for v in self.graph['V'] if v not in self.graph['Y']]

    def U(self, X):
        U = 0
        vars = {**dict(zip(self.X, X)), **self.graph['Y']}
        for v in self.graph['V']:
            d = deterministic_eval(self.graph['P'][v][1], vars)
            U -= d.log_prob(vars[v])
        
        return U

    def grad_U(self, X):
        X.require_grad = True
        U = self.U(X)
        U.backward()

        return X.grad
    
    def H(self, X, R, M):
        X.require_grad = True
        U = self.U(X)
        K = 0.5 * torch.matmul(R.t(), torch.matmul(M.inverse(), R))

        return U + K

    def leapfrog(self, X, R, T, epsilon):
        R_t_half = R - 0.5 * epsilon * self.grad_U(X)
        X_t = copy.deepcopy(X)
        for t in range(T-1):
            X_t = X_t + epsilon * R_t_half
            R_t_half = R_t_half - epsilon * self.grad_U(X_t)
        X_t = X_t + epsilon * R_t_half
        R_t = R_t_half - 0.5 * epsilon * self.grad_U(X_t)

        return X_t, R_t
    
    def hmc(self, X_init, num_samples, T, epsilon, M):
        samples = []
        for i in range(num_samples):
            X = X_init if i == 0 else samples[i-1]
            R = dist.MultivariateNormal(torch.zeros(len(self.X), M))
            X_p, R_p = self.leapfrog(X, R, T, epsilon)
            if torch.rand(1) < torch.exp(-self.H(X_p, R_p, M) + self.H(X, R, M)):
                samples.append(dict(zip(self.X, X_p)))
            else:
                samples.append(dict(zip(self.X, copy.deepcopy(X))))
        
        return samples
    
    def sample(self, num_samples, T, epsilon, M):
        vars = sample_vars_from_joint(self.graph)
        X_init = torch.tensor([vars[x] for x in self.X])
        return self.hmc(X_init, num_samples, T, epsilon, M)
