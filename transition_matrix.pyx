import itertools

import numpy


def increment_transition_counts_from_chain_using_bincount(markov_chain, transition_counts_matrix):
    flat_coords = numpy.ravel_multi_index((markov_chain[:-1], markov_chain[1:]), transition_counts_matrix.shape)
    transition_counts_matrix.flat += numpy.bincount(flat_coords, minlength=transition_counts_matrix.size)

def get_fake_transitions(markov_chains):
    cdef int i, end_of_old, beginning_of_new
    fake_transitions = []
    for i in xrange(1, len(markov_chains)):
        old_chain = markov_chains[i - 1]
        new_chain = markov_chains[i]
        end_of_old = old_chain[-1]
        beginning_of_new = new_chain[0]
        fake_transitions.append((end_of_old, beginning_of_new))
    return fake_transitions

def decrement_fake_transitions(fake_transitions, counts_matrix):
    cdef int old_state, new_state
    for old_state, new_state in fake_transitions:
        counts_matrix[old_state, new_state] -= 1

def chunks(your_list, int n):
    cdef int i
    for i in xrange(0, len(your_list), n):
        yield your_list[i:i + n]

def get_transition_counts_matrix(markov_chains, number_of_states, chains_to_process_at_once=None):
    """Must store 2 additional slice copies of markov chains in memory at once.
    You may need to break up the chains into manageable chunks that don't exceed your memory.
    """
    transition_counts_matrix = numpy.zeros([number_of_states, number_of_states])
    fake_transitions = get_fake_transitions(markov_chains)
    chains_to_process_at_once = chains_to_process_at_once or len(markov_chains)
    for chunk_of_chains in chunks(markov_chains, chains_to_process_at_once):
        joined_markov_chains = list(itertools.chain(*chunk_of_chains))
        increment_transition_counts_from_chain_using_bincount(joined_markov_chains, transition_counts_matrix)
    decrement_fake_transitions(fake_transitions, transition_counts_matrix)
    return transition_counts_matrix

def convert_to_transition_probabilities(transition_counts_matrix, float smoothing_constant=1):
    cdef int number_of_old_states = len(transition_counts_matrix)
    cdef int old_state_number, total_transitions_from_old_state_to_all_new_states
    cdef float denominator_smoothing_factor
    for old_state_number in xrange(0, number_of_old_states):
        total_transitions_from_old_state_to_all_new_states = sum(transition_counts_matrix[old_state_number,])
        denominator_smoothing_factor = number_of_old_states * smoothing_constant
        transition_counts_matrix[old_state_number,] += smoothing_constant
        transition_counts_matrix[old_state_number,] /= total_transitions_from_old_state_to_all_new_states + denominator_smoothing_factor

def get_transition_matrix(markov_chains, number_of_states, chains_to_proces_at_once=None):
    transition_counts_matrix = get_transition_counts_matrix(markov_chains, number_of_states, chains_to_proces_at_once)
    convert_to_transition_probabilities(transition_counts_matrix)
    return transition_counts_matrix

