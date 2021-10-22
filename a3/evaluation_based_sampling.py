import numbers
import torch
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from primitives import funcprimitives
        
def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
    Args:
        ast: json FOPPL program
    Returns: sample from the prior of ast
    """
    
    sigma = {}
    local_env = {}
    func_defs = get_func_defs(ast)
    
    ast = ast[-1]
    ret, sigma = eval(ast, sigma, local_env, func_defs)

    return ret

def eval(e, sigma, local_env, func_defs):
    if (not isinstance(e, list)) or len(e) == 1:
        if isinstance(e, list):
            # empty vector
            if e[0] == 'vector':
                return torch.tensor([]), sigma
            else:
                e = e[0]
        # case c
        if isinstance(e, numbers.Number):
            return torch.tensor(float(e)), sigma
        # case v
        else:
            return local_env[e], sigma
    # case (let [v1 e1] e0)        
    elif e[0] == "let":
        c1, sigma = eval(e[1][1], sigma, local_env, func_defs)
        return eval(e[2], sigma, {**local_env, **{e[1][0]: c1}}, func_defs)
    # case (if e1 e2 e3)
    elif e[0] == "if":
        e1_, sigma = eval(e[1], sigma, local_env, func_defs)
        if e1_:
            return eval(e[2], sigma, local_env, func_defs)
        else:
            return eval(e[3], sigma, local_env, func_defs)
    # case (sample d)
    elif e[0] == "sample":
        dist, sigma = eval(e[1], sigma, local_env, func_defs)
        return dist.sample(), sigma
    # case (observe d y)
    elif e[0] == "observe":
        d1, sigma = eval(e[1], sigma, local_env, func_defs)
        c2, sigma = eval(e[2], sigma, local_env, func_defs)
        sigma['logW'] += d1.log_prob(c2)
        return c2, sigma
    # case (e0 e1 ... en)
    else:
        c = []
        for i in range(1,len(e)):
            ci, sigma = eval(e[i], sigma, local_env, func_defs)
            c.append(ci)
        # case e0 is f
        if e[0] in func_defs:
            return eval(func_defs[e[0]]["exp"], sigma,
                        {**local_env, **dict(zip(func_defs[e[0]]["vars"], c))}, func_defs)
        # case e0 is c
        elif e[0] in funcprimitives:
            return funcprimitives[e[0]](*c), sigma


def get_stream(ast):
    """Return a stream of prior samples"""
    while True:
        yield evaluate_program(ast)
        

def get_func_defs(ast):
    func_defs = {}
    for i in range(len(ast)):
        if isinstance(ast[i], list) and ast[i][0] == "defn":
            func_defs[ast[i][1]] = {"vars": ast[i][2], "exp": ast[i][3]}
    
    return func_defs


def likelihood_weighting(ast, num_samples):
    func_defs = get_func_defs(ast)
    ast = ast[-1]
    samples = []
    log_weights = []

    for i in range(num_samples):
        sigma = {'logW': 0}
        r, sigma = eval(ast, sigma, {}, func_defs)
        samples.append(r)
        log_weights.append(sigma['logW'])

    return samples, log_weights



def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '../a2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate_program(ast)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', '../a2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    

        
if __name__ == '__main__':

    #run_deterministic_tests()
    
    #run_probabilistic_tests()

    
    for i in range(1,5):
        ast = daphne(['desugar', '-i', '../a2/programs/{}.daphne'.format(i)])
        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast))
    