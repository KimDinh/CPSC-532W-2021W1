import numpy as np
from tqdm import tqdm
   
def sample_topic_assignment(topic_assignment,
                            topic_counts,
                            doc_counts,
                            topic_N,
                            doc_N,
                            alpha,
                            gamma,
                            words,
                            document_assignment):
    """
    Sample the topic assignment for each word in the corpus, one at a time.
    
    Args:
        topic_assignment: size n array of topic assignments
        topic_counts: n_topics x alphabet_size array of counts per topic of unique words        
        doc_counts: n_docs x n_topics array of counts per document of unique topics

        topic_N: array of size n_topics count of total words assigned to each topic
        doc_N: array of size n_docs count of total words in each document, minus 1
        
        alpha: prior dirichlet parameter on document specific distributions over topics
        gamma: prior dirichlet parameter on topic specific distribuitons over words.

        words: size n array of wors
        document_assignment: size n array of assignments of words to documents
    Returns:
        topic_assignment: updated topic_assignment array
        topic_counts: updated topic counts array
        doc_counts: updated doc_counts array
        topic_N: updated count of words assigned to each topic
    """
    #TODO
    D = doc_counts.shape[0]
    T = doc_counts.shape[1]
    W = topic_counts.shape[1]
    
    for i, w in enumerate(tqdm(words)):
        d = document_assignment[i]      # the document that current word belongs to
        z = topic_assignment[i]         # topic of the current word

        # decrement counts
        topic_counts[z, w] -= 1
        doc_counts[d, z] -= 1
        topic_N[z] -= 1

        # compute conditional distribution for Gibbs
        p = (doc_counts[d, :] + alpha) * (topic_counts[:, w] + gamma)
        p /= (topic_N + W * gamma) * (doc_N[d] + T * alpha)
        p /= np.sum(p)

        # sampling and update
        z_prime = np.random.choice(T, p=p)
        topic_assignment[i] = z_prime
        topic_counts[z_prime, w] += 1
        doc_counts[d, z_prime] += 1
        topic_N[z_prime] += 1

    return topic_assignment, topic_counts, doc_counts, topic_N