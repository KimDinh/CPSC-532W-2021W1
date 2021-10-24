import torch
import copy
from daphne import daphne
from graph_based_sampling import deterministic_eval, sample_vars_from_joint

class MH_Gibbs:
    """
    MH within Gibbs sampling
    Using pior as proposal distribution
    """

    def __init__(self, graph):
        self.graph = graph[1]
        self.expr = graph[2]
        self.X = [v for v in self.graph['V'] if v not in self.graph['Y']]
        self.graph['Y'] = {k: torch.tensor(v) for k, v in self.graph['Y'].items()}
  
    def accept(self, x, X_p, X):
        q = deterministic_eval(self.graph['P'][x][1], {**X, **self.graph['Y']})
        q_p = deterministic_eval(self.graph['P'][x][1], {**X_p, **self.graph['Y']})
        log_alpha = q_p.log_prob(X[x]) - q.log_prob(X_p[x])

        Vx = [x, *self.graph['A'][x]]
        vars = {**X_p, **self.graph['Y']}
        log_prob_Vx = [deterministic_eval(self.graph['P'][v][1], vars).log_prob(vars[v]) for v in Vx]
        log_alpha += torch.sum(torch.stack(log_prob_Vx))

        vars = {**X, **self.graph['Y']}
        log_prob_Vx = [deterministic_eval(self.graph['P'][v][1], vars).log_prob(vars[v]) for v in Vx]
        log_alpha -= torch.sum(torch.stack(log_prob_Vx))
        
        return torch.exp(log_alpha)

    def gibbs_step(self, X):
        for x in self.X:
            q = deterministic_eval(self.graph['P'][x][1], {**X, **self.graph['Y']})
            X_p = copy.deepcopy(X)
            X_p[x] = q.sample()
            alpha = self.accept(x, X_p, X)
            if torch.rand(1) < alpha:
                X = X_p
        
        return X

    def gibbs(self, X_init, num_samples):
        samples = []
        for i in range(num_samples):
            X = self.gibbs_step(X_init if i == 0 else samples[i-1])
            samples.append(deterministic_eval(self.expr, {**X, **self.graph['Y']}))

        return samples
    
    def sample(self, num_samples):
        vars = sample_vars_from_joint(self.graph)
        X_init = dict((x, vars[x]) for x in self.X)

        return self.gibbs(X_init, num_samples)