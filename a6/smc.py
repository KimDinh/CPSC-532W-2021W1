from evaluator import evaluate
import torch
import numpy as np
import json
import sys
from copy import deepcopy
import gc


def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res

def resample_particles(particles, log_weights):
    n_particles = len(particles)
    weights = np.exp(np.array(log_weights))
    resampled_idx = np.random.choice(range(n_particles), n_particles, p=weights/np.sum(weights))
    new_particles = [particles[i] for i in resampled_idx]
    logZ = np.log(np.sum(weights) / n_particles)
    
    return logZ, new_particles



def SMC(n_particles, exp):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):

        res = evaluate(exp, env=None)('addr_start', output)
        logW = 0.


        particles.append(res)
        weights.append(logW)

    #can't be done after the first step, under the address transform, so this should be fine:
    done = False
    smc_cnter = 0
    while not done:
        #print('In SMC step {}, Zs: '.format(smc_cnter), logZs)
        for i in range(n_particles): #Even though this can be parallelized, we run it serially
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res[0]
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                #TODO: check particle addresses, and get weights and continuations
                particles[i] = res
                sigma = res[2]
                weights[i] = sigma['logW'].detach()

                if i == 0:
                    alpha_cur = sigma['alpha']
                else:
                    assert(alpha_cur == sigma['alpha'])
        
        if not done:
            #resample and keep track of logZs
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)
        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles


if __name__ == '__main__':

    for i in range(1,5):
        with open('programs/{}.json'.format(i),'r') as f:
            exp = json.load(f)
        n_particles = None #TODO 
        logZ, particles = SMC(n_particles, exp)

        print('logZ: ', logZ)

        values = torch.stack(particles)
        #TODO: some presentation of the results
