import torch
import torch.distributions as dist
from collections import deque

from daphne import daphne

from primitives import funcprimitives #TODO
from tests import is_tol, run_prob_test,load_truth

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
env = {
    **funcprimitives,
    'sample*': lambda d: d.sample(),
    'observe*': lambda d, y: y,
    'if': lambda e1, e2, e3: e2 if e1 else e3,
}


def deterministic_eval(exp, vars):
    "Evaluation function for the deterministic target language of the graph based representation."
    if type(exp) is list:
        op = exp[0]
        args = exp[1:]
        return env[op](*[deterministic_eval(arg, vars) for arg in args])
    elif type(exp) in [int, float, bool]:
        # We use torch for all numerical objects in our evaluator
        return torch.tensor(float(exp))
    elif exp in vars:
        return vars[exp]
    elif exp is None:
        return None
    else:
        raise("Expression type unknown.", exp)

def topological_sort(vertices, edges):
    in_deg = dict(zip(vertices, [0]*len(vertices)))
    for u in vertices:
        if u in edges:
            for v in edges[u]:
                in_deg[v] += 1

    topo_order = []
    q = deque([u for u in vertices if in_deg[u] == 0])
    while True:
        try:
            u = q.popleft()
            topo_order.append(u)
            if u in edges:
                for v in edges[u]:
                    in_deg[v] -= 1
                    if in_deg[v] == 0:
                        q.append(v)
        except:
            break
    
    return topo_order

def sample_from_joint(graph):
    "This function does ancestral sampling starting from the prior."
    vars = sample_vars_from_joint(graph[1])

    return deterministic_eval(graph[2], vars)


def sample_vars_from_joint(graph):
    topo_order = topological_sort(graph['V'], graph['A'])
    vars = {}
    for u in topo_order:
        vars[u] = deterministic_eval(graph['P'][u], vars)
    
    return vars


def log_joint_density(graph, vars):
    "Return the log joint density given the graph and values of the variables"
    log_joint = torch.tensor(0.0)
    for u in graph['V']:
        link = deterministic_eval(graph['P'][u][1], vars)
        if type(vars[u]) in [int, float, bool]:
            val = torch.tensor(float(vars[u])) 
        else:
            val = vars[u]
        log_joint += link.log_prob(val)

    return log_joint


def get_stream(graph):
    """Return a stream of prior samples
    Args: 
        graph: json graph as loaded by daphne wrapper
    Returns: a python iterator with an infinite stream of samples
        """
    while True:
        yield sample_from_joint(graph)




#Testing:

def run_deterministic_tests():
    
    for i in range(1,13):
        #note: this path should be with respect to the daphne path!
        graph = daphne(['graph','-i','../a3/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = deterministic_eval(graph[-1], {})
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    #TODO: 
    num_samples = 1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        graph = daphne(['graph', '-i', '../a3/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(graph)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    


        
        
if __name__ == '__main__':
    

    #run_deterministic_tests()
    #run_probabilistic_tests()

    for i in range(1,5):
        graph = daphne(['graph','-i','../a3/programs/{}.daphne'.format(i)])
        print(graph, "\n\n\n")
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(sample_from_joint(graph))    

    