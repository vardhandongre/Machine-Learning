def max_single_step(previous_log_probs, transition_probs, emission_prob, y):
    """
  Args:
    previous_log_probs: a numpy array with shape (S,).  previous_log_probs[i] is the initial probability of state Z_i at time t-1.
    transition_probs: a numpy array with shape (S, S).  transition_probs[i,j] is the probability of transitioning from Z_j to Z_i.
    emission_prob: a numpy array with shape (N, S).  emission_probs[i,j] is the probability of emission Y_i given state Z_j.
    y: an integer in [0,N), denoting the emission observed
  Returns:
    A tuple (pz, path), where
      pz is a numpy array of shape (S,), where pz[i] stores the log joint probability of the most likely path so far that led to hidden state Z_i and emission y at time t.
      path is a numpy int array of shape (S,), where path[i] is the most likely state that preceded observing Z_i at time t.
  """
    states = previous_log_probs.shape[0]
    N = len(emission_prob)
    pz = np.zeros(states)
    p_c = {}
    path = []
    for s in states:
        p_c = np.log(previous_log_probs[s]) + np.log(emission_prob[y[0],s])
    for i in range(1,len(y)):
        pre_prob = p_c
        p_c = {}
        for s_c in states:
            max_prob, last = np.max((np.log(pre_prob[s_pre]*transition_probs[s_pre][s_c]*emission_prob[s_c][y[i]]), s_pre) for s_pre in states)
            p_c[s_c] = max_prob
            path[s_c].append(last)
    return (p_c, path)


def Viterbit(obs, states, s_pro, t_pro, e_pro):
    path = { s:[] for s in states} # init path: path[s] represents the path ends with s
    curr_pro = {}
    for s in states:
        curr_pro[s] = s_pro[s]*e_pro[s][obs[0]]
    for i in xrange(1, len(obs)):
        last_pro = curr_pro
        curr_pro = {}
        for curr_state in states:
            max_pro, last_sta = max(((last_pro[last_state]*t_pro[last_state][curr_state]*e_pro[curr_state][obs[i]], last_state) 
                                       for last_state in states))
            curr_pro[curr_state] = max_pro
            path[curr_state].append(last_sta)

    # find the final largest probability
    max_pro = -1
    max_path = None
    for s in states:
        path[s].append(s)
        if curr_pro[s] > max_pro:
            max_path = path[s]
            max_pro = curr_pro[s]
        # print '%s: %s'%(curr_pro[s], path[s]) # different path and their probability
    return max_path