import torch
import operator as op
import copy
import torch.distributions as dist

def vector(*args):
    try:
        return torch.stack(list(args))
    except:
        return list(args)

def hashmap(*args):
    e = list(args)
    # tensor.item() to get the value in the tensor
    return dict(zip([e_.item() for e_ in e[0::2]], e[1::2]))

def get(e1, e2):
    if isinstance(e1, dict):
        return e1[e2.item()]
    else:
        return e1[int(e2)]

def append(e1, e2):
    ret = copy.deepcopy(e1)
    if isinstance(ret, torch.Tensor):
        ret = torch.cat([ret, e2.unsqueeze(dim=0)])
    else:
        ret.append(e2)
    return ret

def put(e1, e2, e3):
    ret = copy.deepcopy(e1)
    if isinstance(ret, dict):
        ret[e2.item()] = e3
    else:
        ret[int(e2)] = e3
    return ret

def cons(e1, e2):
    ret = copy.deepcopy(e2)
    if isinstance(ret, torch.Tensor):
        ret = torch.cat([e1.unsqueeze(dim=0), ret])
    else:
        ret.insert(0, e1)
    return ret


funcprimitives = {
    '+': op.add,
    '-': op.sub,
    '*': op.mul,
    '/': op.truediv,
    'sqrt': lambda e: torch.sqrt(e),
    'exp': lambda e: torch.exp(e),
    '>': op.gt,
    '<': op.lt,
    '>=': op.ge,
    '<=': op.le,
    '=': op.eq,
    'vector': vector,
    'hash-map': hashmap,
    'get': get,
    'first': lambda e: get(e, 0),
    'second': lambda e: get(e, 1),
    'last': lambda e: get(e, -1),
    'rest': lambda e: e[1:],
    'append': append,
    'put': put,
    'cons': cons,
    'conj': append,
    'mat-mul': torch.matmul,
    'mat-add': op.add,
    'mat-transpose': lambda e: e.T,
    'mat-tanh': torch.tanh,
    'mat-repmat': lambda e1, e2, e3: e1.repeat(int(e2), int(e3)),
    'normal': dist.Normal,
    'uniform': dist.Uniform,
    'beta': dist.Beta,
    'bernoulli': dist.Bernoulli,
    'exponential': dist.Exponential,
    'discrete': dist.Categorical
}