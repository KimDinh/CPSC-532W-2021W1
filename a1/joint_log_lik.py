import numpy as np
from scipy.special import loggamma

def joint_log_lik(doc_counts, topic_counts, alpha, gamma):
    """
    Calculate the joint log likelihood of the model
    
    Args:
        doc_counts: n_docs x n_topics array of counts per document of unique topics
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.
    Returns:
        jll: the joint log likelihood of the model
    """
    #TODO
    jll = 0
    D = doc_counts.shape[0]
    T = doc_counts.shape[1]
    W = topic_counts.shape[1]

    jll += D * loggamma(T * alpha) - D * T * loggamma(alpha)
    jll += np.sum(np.sum(loggamma(doc_counts + alpha), axis=1) - 
            loggamma(np.sum(doc_counts + alpha, axis=1)))
    jll += T * loggamma(W * gamma) - T * W * loggamma(gamma)
    jll += np.sum(np.sum(loggamma(topic_counts + gamma), axis=1) - 
            loggamma(np.sum(topic_counts + gamma, axis=1)))
    
    return jll