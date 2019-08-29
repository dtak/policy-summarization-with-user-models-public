"""
Find the value function associated with a policy. Based on Sutton & Barto, 1998.

Adapted from: Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np

def value(policy, n_states, transition_probabilities, reward, discount,
                    threshold=1e-2):
    """
    Find the value function associated with a policy.

    policy: List of action ints for each state.
    n_states: Number of states. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    -> Array of values for each state
    """
    v = np.zeros(n_states)

    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            vs = v[s]
            a = policy[s]
            v[s] = sum(transition_probabilities[s, a, k] *
                       (reward[k] + discount * v[k])
                       for k in range(n_states))
            diff = max(diff, abs(vs - v[s]))

    return v

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2, valid_actions={}, 
                  consider_valid_only=False):
    """
    Find the optimal value function.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    valid actions: dictionairy of valid actions per state, terminal states have no valid actions.
    consider_valid_only: if true consider only valid actions when calculating value.
    -> Array of values for each state
    """

    v = np.zeros(n_states)

    actions = range(n_actions)

    count = 0
    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            max_v = float("-inf")

            if consider_valid_only:
                # Get state valid actions and calculate values only based on them
                state_valid_actions = valid_actions[s]
                actions = state_valid_actions
                # Terminal state has no valid actions, all actions lead back to state. calculate value based on first action
                if not state_valid_actions:
                    state_valid_actions.append(0)

            for a in actions:
                tp = transition_probabilities[s, a, :]
                max_v = max(max_v, np.dot(tp, reward + discount*v)) 

            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v

    return v

def find_policy(n_states, n_actions, transition_probabilities, reward, discount,
                threshold=1e-2, v=None, stochastic=True, valid_actions={}, consider_valid_only=False,
                return_multiple=False):
    """
    Find the optimal policy.

    n_states: Number of states. int.
    n_actions: Number of actions. int.
    transition_probabilities: Function taking (state, action, state) to
        transition probabilities.
    reward: Vector of rewards for each state.
    discount: MDP discount factor. float.
    threshold: Convergence threshold, default 1e-2. float.
    v: Value function (if known). Default None.
    stochastic: Whether the policy should be stochastic. Default True.
    valid actions: dictionairy of valid actions per state, terminal states have no valid actions.
    consider_valid_only: if true consider only valid actions when calculating q-values
    -> Action probabilities for each state or action int for each state
        (depending on stochasticity).
    """

    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold, valid_actions=valid_actions, 
                          consider_valid_only=consider_valid_only)

    if stochastic:
        actions = range(n_actions)
        # Get Q using equation 9.2 from Ziebart's thesis.
        Q = np.zeros((n_states, n_actions))
        for i in range(n_states):

            if consider_valid_only:
                # Get state valid actions (terminal state has no valid actions - qvalues=0)
                state_valid_actions = valid_actions[i]
                actions = state_valid_actions

            for j in actions:
                p = transition_probabilities[i, j, :]
                Q[i, j] = p.dot(reward + discount*v)
        Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
        Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))

        return Q

    def _policy(s):

        if not valid_actions:
            possible_actions = range(n_actions)
        else:
            if not valid_actions[s]:
                # terminal state - no valid actions, return first action
                return 0
            else:
                possible_actions = valid_actions[s]

        vals = []
        for a in possible_actions:
            vals.append( sum(transition_probabilities[s, a, k] *
                                     (reward[k] + discount * v[k])
                                     for k in range(n_states)))
        max_val = max( vals )

        multiple = False
        if len( np.where( ( np.abs( max_val - vals ) ) < 0.1 )[ 0 ] ) > 1:
            multiple = True

        return possible_actions[ np.random.choice( np.where( vals == np.max( vals ) )[ 0 ] ) ] , multiple

    policy = np.array([_policy(s)[0] for s in range(n_states)])

    ##Get indices of states with multiple optimal actions
    multiple_array = np.array([_policy(s)[1] for s in range(n_states)])
    multiple_state_indices = np.where(multiple_array == True)[0]

    if return_multiple:
        return policy , multiple_state_indices
    else:
        return policy
