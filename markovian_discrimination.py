"""Scores used for binary classification based on sequential state information
Algorithm described in:
Predicting home-appliance acquisition sequences: 
Markov/Markov for Discrimination and survival analysis for modeling sequential information in NPTB models
Decision Support Systems, Volume 44, Issue 1, November 2007, Pages 28-45
Anita Prinzie, Dirk Van den Poel

http://www.sciencedirect.com/science/article/pii/S0167923607000334

For maximum performance, use logit scores instead of probabilities (saves resources on conversions)
"""

import numpy

import pyximport; pyximport.install();
import transition_matrix


def get_logit_score_from_markov_chain(markov_chain, positive_transition_matrix, negative_transition_matrix):
    logit_score = 0
    for i in xrange(1, len(markov_chain)):
        old_state = markov_chain[i - 1]
        new_state = markov_chain[i]
        positive_probability = positive_transition_matrix[old_state, new_state]
        negative_probability = negative_transition_matrix[old_state, new_state]
        logit_score += numpy.log(positive_probability / negative_probability)
    return logit_score

def get_mfd_logit_scores(markov_chains, positive_transition_matrix, negative_transition_matrix):
    logit_scores = numpy.zeros(len(markov_chains))
    for i in xrange(0, len(markov_chains)):
        logit_scores[i] = get_logit_score_from_markov_chain(markov_chains[i], positive_transition_matrix, negative_transition_matrix)
    return logit_scores

def logit_score_to_probability(logit_score):
    return numpy.exp(logit_score) / (1 + numpy.exp(logit_score))

def get_probability_scores_from_logit_scores(logit_scores):
    return numpy.array([logit_score_to_probability(logit_score) for logit_score in logit_scores])


def demo():
    """Creates transition matrices for 2 pre-determined classes of customer: churner and non-churner
    Churner: someone who ended his/her subscription
    Non-churner: someone who did not 
    
    Takes some Markov chains for 4 churners and non-churners, 
    and uses them to create transition matrices.
    
    Then, based on those transition matrices,  
    several customers whose classes are unknown are scored from their Markov chains.  
    
    States:
    0: Poor customer loyalty
    1: Medium loyalty
    2: High loyalty
    
    2.4GHz processor takes about 1.6 seconds 
    to calculate transition matrices from 400,000 Markov chains that contain about 1MM total state transitions
    """
    markov_chains_nonchurn = []
    markov_chains_nonchurn.append([0,1,2,1,1,0,2])
    markov_chains_nonchurn.append([0,1,1,1])
    markov_chains_nonchurn.append([0,1,2,2])
    markov_chains_nonchurn.append([2,1,2,2,2,2])
    markov_chains_nonchurn *= 50000
    
    markov_chains_churn = []
    markov_chains_churn.append([2,1,2,1,1,0,0])
    markov_chains_churn.append([2,1,1,0])
    markov_chains_churn.append([2,1,0])
    markov_chains_churn.append([2,1,0,0,0,0])
    markov_chains_churn *= 50000
    
    nonchurn_transition_matrix = transition_matrix.get_transition_matrix(markov_chains_nonchurn, 3)
    churn_transition_matrix = transition_matrix.get_transition_matrix(markov_chains_churn, 3)
    print 'Churn transition probabilities matrix:'
    print churn_transition_matrix
    print 'Non-churn transition probabilities matrix:'
    print nonchurn_transition_matrix
    print
    
    chains_to_score = [numpy.array([2,1,0]), numpy.array([0,1,2]), [0,1,2], [1,2,1,1,2]]
    logit_scores = get_mfd_logit_scores(chains_to_score, churn_transition_matrix, nonchurn_transition_matrix)
    #probability scores = the probability that each chain belongs to the positive (churn) group
    probability_scores = get_probability_scores_from_logit_scores(logit_scores)
    print 'Probabilities of being a churner:'
    print probability_scores

if __name__ == '__main__':
    demo()