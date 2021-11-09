from primitives import env as penv
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from pyrsistent import pmap, plist, pvector
import numbers, torch

def standard_env():
    "An environment with some Scheme standard procedures."
    env = pmap(penv)
    env = env.update({'alpha' : ''}) 

    return env


class Env(dict):
    """
    Copied from https://norvig.com/lispy.html and modified
    """
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer

    def get(self, var):
        return self[var] if (var in self) else self.outer.get(var)


class Procedure(object):
    """
    Copied from https://norvig.com/lispy.html
    """
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env
    
    def __call__(self, *args, sigma): 
        return eval(self.body, sigma, Env(self.parms, args, self.env))


def evaluate(exp):
    func, sigma = eval(exp)
    ret, sigma = func(sigma=sigma)
    return ret


def eval(e, sigma={}, env=None):

    if env is None:
        env = standard_env()

    if (not isinstance(e, list)):
        if isinstance(e, list):
            e = e[0]
        if isinstance(e, numbers.Number):
            return torch.tensor(float(e)), sigma
        elif isinstance(e, str):
            val = env.get(e)
            if val is not None:
                return val, sigma
            else:
                return e, sigma
        else:
            raise("Expression type unknown.", e) 
    else:
        op = e[0]
        args = e[1:]
        if op == 'if':
            pred, sigma = eval(args[0], sigma, env)
            if pred:
                return eval(args[1], sigma, env)
            else:
                return eval(args[2], sigma, env)
        elif op == 'fn':
            params, body = args
            return Procedure(params, body, env), sigma
        elif op == 'sample':
            # evaluate push-address
            _, sigma = eval(args[0], sigma, env)
            # evaluate distribution and sample
            d, sigma = eval(args[1], sigma, env)
            return d.sample(), sigma
        elif op == 'observe':
            # evaluate push-address
            _, sigma = eval(args[0], sigma, env)
            # evaluate distribution and observed value
            d, sigma = eval(args[1], sigma, env)
            c, sigma = eval(args[2], sigma, env)
            return c, sigma
        else:
            func, sigma = eval(op, sigma, env)
            vals = []
            for arg in args:
                c, sigma = eval(arg, sigma, env)
                vals.append(c)
            if isinstance(func, Procedure):
                return func(*vals, sigma=sigma)
            else:
                return func(*vals), sigma


def get_stream(exp):
    while True:
        yield evaluate(exp)


def run_deterministic_tests():
    
    for i in range(1,14):

        exp = daphne(['desugar-hoppl', '-i', '../a5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('FOPPL Tests passed')
    
    for i in range(1,13):

        exp = daphne(['desugar-hoppl', '-i', '../a5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-2
    
    for i in range(1,7):
        exp = daphne(['desugar-hoppl', '-i', '../a5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
        
    print('All probabilistic tests passed')    



if __name__ == '__main__':
    
    #run_deterministic_tests()
    #run_probabilistic_tests()
    
    for i in range(1,4):
        exp = daphne(['desugar-hoppl', '-i', '../a5/programs/{}.daphne'.format(i)])
        print('Sample of prior of program {}:'.format(i))
        print(evaluate(exp), "\n\n\n")      
