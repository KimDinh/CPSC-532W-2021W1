import torch
from graph_based_sampling import topological_sort, deterministic_eval
import distributions
from tqdm import tqdm

optim_dist = {
    'normal': distributions.Normal,
    'bernoulli': distributions.Bernoulli,
    'discrete': distributions.Categorical,
    'gamma': distributions.Gamma,
    'dirichlet': distributions.Dirichlet,
    'uniform-continuous': lambda low, high: distributions.Gamma((low+high)/2, torch.tensor(1.0))
}

class BBVI:
    def __init__(self, graph, lr=1e-3):
        self.graph = graph[1]
        self.expr = graph[2]
        self.X = [v for v in self.graph['V'] if v not in self.graph['Y']]
        self.graph['Y'] = {k: torch.tensor(float(v)) for k, v in self.graph['Y'].items()}
        self.topo_order = topological_sort(self.graph['V'], self.graph['A'])
        self.optimizer = torch.optim.Adam([torch.tensor(0.0)], lr=lr)
        self.init_Q()   # initialize proposal distributions
        self.optimizer.zero_grad()
    
    def init_Q(self):
        self.Q = {}
        vars = {**{x: torch.tensor(0.0) for x in self.X}, **self.graph['Y']}
        for v in self.topo_order:
            if v in self.X:
                dist_type = self.graph['P'][v][1][0]
                dist_param = [deterministic_eval(e, vars) for e in self.graph['P'][v][1][1:]]
                self.Q[v] = optim_dist[dist_type](*dist_param).make_copy_with_grads()
                self.optimizer.param_groups[0]['params'] += self.Q[v].Parameters()
                vars[v] = deterministic_eval(self.graph['P'][v], vars)
        
    def bbvi_eval(self, sigma, vars):
        for v in self.topo_order:
            d = deterministic_eval(self.graph['P'][v][1], vars)
            if v in self.X:
                # sample evaluation
                with torch.no_grad():
                    c = self.Q[v].sample()
                log_prob_q = self.Q[v].log_prob(c)
                log_prob_q.backward()
                sigma['grads'][v] = [p.grad.clone().detach() for p in self.Q[v].Parameters()]
                sigma['logW'] += (d.log_prob(c.detach()) - log_prob_q.clone().detach())
                vars[v] = c.clone().detach()
            else:
                # observe evaluation
                vars[v] = self.graph['Y'][v]
                sigma['logW'] += d.log_prob(vars[v]).clone().detach()
        
        return deterministic_eval(self.expr, vars), sigma

    def optimizer_step(self, elbo_grads):
        for v in self.X:
            for d, param in enumerate(self.Q[v].Parameters()):
                param.grad = -elbo_grads[v][d]
        
        self.optimizer.step()
        self.optimizer.zero_grad()


    def elbo_gradients(self, grads, log_weights):
        elbo_grads = {v: [] for v in self.X}
        L = len(grads)
        for v in self.X:
            D = len(grads[0][v])
            param_to_G = {d: [] for d in range(D)}  # map from param d to L gradients w.r.t param d
            param_to_F = {d: [] for d in range(D)}
            for l in range(L):
                for d, param in enumerate(grads[l][v]):
                    param_to_G[d].append(param)
                    param_to_F[d].append(param*log_weights[l])
            
            for d in range(D):
                # convert to tensor of size (L, K) where K is the length of param d
                if param_to_G[d][0].dim() == 0:
                    param_to_G[d] = torch.tensor(param_to_G[d]).unsqueeze(1)
                    param_to_F[d] = torch.tensor(param_to_F[d]).unsqueeze(1)
                else:
                    param_to_G[d] = torch.stack(param_to_G[d], dim=0)
                    param_to_F[d] = torch.stack(param_to_F[d], dim=0)
                
                K = param_to_G[d].size()[1]     # length of parameter d
                # computing baseline
                cov_sum, var_sum = torch.tensor(0.0), torch.tensor(0.0)
                for k in range(K):
                    FG_mat = torch.stack((param_to_F[d][:, k], param_to_G[d][:, k]), dim=0)
                    cov_mat = torch.cov(FG_mat, correction=0)
                    cov_sum += cov_mat[0, 1]
                    var_sum += cov_mat[1, 1]
                
                g = torch.sum(param_to_F[d] - cov_sum/var_sum * param_to_G[d], dim=0) / L
                elbo_grads[v].append(g.squeeze(0) if K == 1 else g)
    
        return elbo_grads
                
    def bbvi(self, T, L):
        rets = []
        log_weights = []
        grads = []
        elbos = []
        pbar = tqdm(range(T))
        for t in pbar:
            for l in range(L):
                sigma = {'logW': torch.tensor(0.0), 'grads': {}}
                vars = {}
                r, sigma = self.bbvi_eval(sigma, vars)
                self.optimizer.zero_grad()
                grads.append(sigma['grads'])
                log_weights.append(sigma['logW'])
                rets.append(r)
            
            elbos.append(sum(log_weights[-L:])/L)
            elbo_grads = self.elbo_gradients(grads[-L:], log_weights[-L:])
            self.optimizer_step(elbo_grads)
            pbar.set_postfix({'ELBO': elbos[-1]})
        
        return rets, log_weights, elbos

    def sample(self, T, L):
        return self.bbvi(T, L)