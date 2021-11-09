import torch
import torch.distributions as dist
import copy


class Normal(dist.Normal):
    
    def __init__(self, alpha, loc, scale):
        
        if scale > 20.:
            self.optim_scale = scale.clone().detach().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()
        
        
        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]
         
        return Normal(*ps)
    
    def log_prob(self, x):
        
        self.scale = torch.nn.functional.softplus(self.optim_scale)
        
        return super().log_prob(x)
        

def push_addr(alpha, value):
    return alpha + value

def vector(addr, *args):
    if len(args) == 0:
        return torch.Tensor([])
    try:
        return torch.stack(list(args))
    except:
        return list(args)

def hashmap(addr, *args):
    e = list(args)
    keys = [e_.item() if isinstance(e_, torch.Tensor) else e_ for e_ in e[0::2]]
    return dict(zip(keys, e[1::2]))

def get(addr, e1, e2):
    if isinstance(e1, dict):
        return e1[e2.item() if isinstance(e2, torch.Tensor) else e2]
    else:
        return e1[int(e2)]

def put(addr, e1, e2, e3):
    ret = copy.deepcopy(e1)
    if isinstance(ret, dict):
        ret[e2.item() if isinstance(e2, torch.Tensor) else e2] = e3
    else:
        ret[int(e2)] = e3
    return ret

def append(addr, e1, e2):
    ret = copy.deepcopy(e1)
    if isinstance(ret, torch.Tensor):
        ret = torch.cat([ret, e2.unsqueeze(dim=0)])
    else:
        ret.append(e2)
    return ret

def cons(addr, e1, e2):
    ret = copy.deepcopy(e1)
    if isinstance(ret, torch.Tensor):
        ret = torch.cat([e2.unsqueeze(dim=0), ret])
    else:
        ret.insert(0, e2)
    return ret

def empty(addr, e):
    if isinstance(e, torch.Tensor):
        return e.size()[0] == 0
    else:
        return len(e) == 0

env = {
    'normal': Normal,
    'uniform-continuous': lambda addr, e1, e2: torch.distributions.Uniform(e1, e2),
    'beta': lambda addr, e1, e2: torch.distributions.Beta(e1, e2),
    'exponential': lambda addr, e: torch.distributions.Exponential(e),
    'discrete': lambda addr, e: torch.distributions.Categorical(e),
    'gamma': lambda addr, e1, e2: torch.distributions.Gamma(e1, e2),
    'dirichlet': lambda addr, e: torch.distributions.Dirichlet(e),
    'flip': lambda addr, e: torch.distributions.Bernoulli(e),
    'push-address' : push_addr,
    '+': lambda addr, e1, e2: e1 + e2,
    '-': lambda addr, e1, e2: e1 - e2,
    '*': lambda addr, e1, e2: e1 * e2,
    '/': lambda addr, e1, e2: e1 / e2,
    'sqrt': lambda addr, e: torch.sqrt(e),
    'exp': lambda addr, e: torch.exp(e),
    'log': lambda addr, e: torch.log(e),
    'abs': lambda addr, e: torch.abs(e),
    '>': lambda addr, e1, e2: e1 > e2,
    '<': lambda addr, e1, e2: e1 < e2,
    '>=': lambda addr, e1, e2: e1 >= e2,
    '<=': lambda addr, e1, e2: e1 <= e2,
    '=': lambda addr, e1, e2: e1 == e2,
    'and': lambda addr, e1, e2: e1 and e2,
    'or': lambda addr, e1, e2: e1 or e2,
    'vector': vector,
    'hash-map': hashmap,
    'get': get,
    'first': lambda addr, e: get(addr, e, 0),
    'second': lambda addr, e: get(addr, e, 1),
    'last': lambda addr, e: get(addr, e, -1),
    'rest': lambda addr, e: e[1:],
    'peek': lambda addr, e: get(addr, e, -1),
    'append': append,
    'put': put,
    'conj': append,
    'cons': cons,
    'empty?': empty
}






