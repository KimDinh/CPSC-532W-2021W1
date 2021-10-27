import torch
import copy
from daphne import daphne
import torch.distributions as dist
from graph_based_sampling import deterministic_eval, sample_vars_from_joint, log_joint_density

class HMC:
    def __init__(self, graph):
        self.graph = graph[1]
        self.expr = graph[2]
        self.X = [v for v in self.graph['V'] if v not in self.graph['Y']]
        self.graph['Y'] = {k: torch.tensor(float(v)) for k, v in self.graph['Y'].items()}

    def U(self, X):
        U = 0
        vars = {**dict(zip(self.X, X)), **self.graph['Y']}
        for v in self.graph['V']:
            d = deterministic_eval(self.graph['P'][v][1], vars)
            U -= d.log_prob(vars[v])

        return U

    def grad_U(self, X):
        X.requires_grad_(True)
        U = self.U(X)
        U.backward()

        return X.grad
    
    def H(self, X, R, M):
        X.requires_grad_(True)
        U = self.U(X)
        K = 0.5 * torch.matmul(R.t(), torch.matmul(M.inverse(), R))

        return U + K

    def leapfrog(self, X, R, T, epsilon):
        R_t = R - 0.5 * epsilon * self.grad_U(X)
        X_t = X.detach().clone()
        for t in range(T-1):
            X_t = X_t.detach() + epsilon * R_t
            R_t = R_t - epsilon * self.grad_U(X_t)
        X_t = X_t.detach() + epsilon * R_t
        R_t = R_t - 0.5 * epsilon * self.grad_U(X_t)

        return X_t.detach(), R_t
    
    def hmc(self, X_init, num_samples, T, epsilon, M):
        R_dist = dist.MultivariateNormal(torch.zeros(len(self.X)), M)
        samples = []
        log_density = []
        X = X_init
        for i in range(num_samples+1000):
            R = R_dist.sample()
            X_p, R_p = self.leapfrog(X, R, T, epsilon)
            if torch.rand(1) < torch.exp(-self.H(X_p, R_p, M) + self.H(X, R, M)):
                if i >= 1000:
                    vars = {**dict(zip(self.X, X_p)), **self.graph['Y']}
                    samples.append(deterministic_eval(self.expr, vars).detach())
                    log_density.append(log_joint_density(self.graph, vars).detach())
                X = X_p
            else:
                if i >= 1000:
                    vars = {**dict(zip(self.X, X)), **self.graph['Y']}
                    samples.append(deterministic_eval(self.expr, vars).detach())
                    log_density.append(log_joint_density(self.graph, vars).detach())

        return samples, log_density
    
    def sample(self, num_samples, T, epsilon, M):
        vars = sample_vars_from_joint(self.graph)
        X_init = torch.tensor([vars[x] for x in self.X])
        return self.hmc(X_init, num_samples, T, epsilon, M)
