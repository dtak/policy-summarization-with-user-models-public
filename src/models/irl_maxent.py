"""
Implements maximum entropy inverse reinforcement learning (Ziebart et al., 2008)

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""
from itertools import product
import numpy as np
import numpy.random as rn
from value_iteration import find_policy
from threading import Thread, Lock
import pylab as plt
from tqdm import tqdm

class Maxent:

    def __init__(self, discount=0.9, epochs=300, learning_rate=1e-2, gd_threshold=1e-2, mc_rollouts=0, rollout_horizon=20, par=True, verbose=True):

        ##Gradient descent and MDP parameters
        self.discount = discount
        self.epochs = epochs  ##Number of GD epochs
        self.learning_rate = learning_rate  ##GD learning rate
        self.gd_threshold = gd_threshold

        ##Expected svf approximation parms
        self.mc_rollouts = mc_rollouts  ##Number of MC rollouts (set to 0 for direct calculatio without MC)
        self.rollout_horizon = rollout_horizon

        ##Multithread or not
        self.par = par
        self.verbose = verbose ##Print or not

    ##Calculate reward function based on given trajectories
    def irl(self, n_actions, transition_probability, feature_matrix, trajectories, valid_actions={}):

        ##MDP state and action data
        self.n_actions = n_actions
        self.transition_probability = transition_probability
        self.valid_actions = valid_actions
        self.feature_matrix = feature_matrix
        self.n_states, self.d_states = feature_matrix.shape

        ##Trajectories from original policy
        self.trajectories = trajectories

        ##Initialise weights.
        alpha = rn.uniform(size=(self.d_states,))

        ##Calculate the feature expectations \tilde{phi}
        feature_expectations, norm_feature_expectations, feature_counts_by_trajectory = self.find_feature_expectations()

        ##Calculate start state frequencies based on trajectories
        p_start_state = self.find_start_state_freq()

        grads = [] ##Plot gradients
        ##Gradient descent on alpha
        diff = 0
        ##Run for epoch iterations or util diff<gd_threshold
        for i in tqdm(range(self.epochs)):
        #for i in range(self.epochs):

            r = self.feature_matrix.dot(alpha)

            ##Check changes in reward vector
            if i > 0:
                diff = np.max(abs(last_r - r))
            if diff <= self.gd_threshold and i > 0:
                ##No significant changes - finish gd
                break
                #pass

            else:
                ##Significant changes in reward vector - calculate another iteration

                if self.mc_rollouts > 0:
                    ##Approximate expected svf with MC Rollouts
                    expected_svf, q_values = self.find_expected_svf_MC(r, p_start_state)
                else:
                    ##Calculate expected svf directly for all state action pairs
                    expected_svf, q_values = self.find_expected_svf(r, p_start_state)

                ##Normalize expected svf to sum to 1
                norm_expected_svf = expected_svf / expected_svf.sum()

                grad = norm_feature_expectations - self.feature_matrix.T.dot(norm_expected_svf) #eq. 6 in paper
                #diff = np.abs( grad ).sum()

                alpha += ( self.learning_rate * grad )

                ##Save reward vector from last iteration
                last_r = np.copy(r)

                grads.append( np.abs( grad ).sum() )

        return self.feature_matrix.dot(alpha).reshape((self.n_states,)), q_values, self.feature_matrix.T.dot(norm_expected_svf)


    ##Feature expectations for given trajectories - average path feature vector
    def find_feature_expectations(self):

        ##array of feature expectations- each entry represents a feature
        sum_feature_expectations = np.zeros(self.feature_matrix.shape[1])

        feature_counts_by_trajectory = [] ##Feature counts sum for each trajectory
        n_steps = 0

        for trajectory in self.trajectories:
            trajectory_feature_counts = np.zeros(self.feature_matrix.shape[1])
            for state, _ in trajectory:
                sum_feature_expectations += self.feature_matrix[state]
                trajectory_feature_counts += self.feature_matrix[state]
                n_steps += 1

            feature_counts_by_trajectory.append(trajectory_feature_counts)

        ##Average trajectory feature expectations - divide sum by num of trajectories (original definition in paper)
        avg_feature_expectations =  sum_feature_expectations / self.trajectories.shape[0]

        ##Normalized feature expectations - divide sum by total num of steps in trajectories
        norm_feature_expectations = sum_feature_expectations / n_steps

        return avg_feature_expectations, norm_feature_expectations, np.array(feature_counts_by_trajectory)

    ##Calculate expected visitation frequencies for all states with MC simulations based on current policy
    def find_expected_svf_MC(self, r, p_start_state):

        ##Calculates probability for each action and state - stochastic policy (lines 1-3 alg 1)
        policy = find_policy(self.n_states, self.n_actions,self.transition_probability, r,
                             self.discount, valid_actions=self.valid_actions, consider_valid_only=True)

        ##Initialize svf matrix and lock
        expected_svf = np.zeros((self.n_states, self.rollout_horizon))
        expected_svf[:, 0] = p_start_state

        if self.par:
            lock_expected_svf = Lock()

        ##Run MC rollouts parallely to calculate state visitation frequencies (run 100 threads parllely)
        rollouts = []
        for j in range(0, self.mc_rollouts, 100):
            if self.par:
                par_rollouts = 100 if self.mc_rollouts - j > 100 else self.mc_rollouts - j
                rollouts = []
                for i in range(par_rollouts):
                    rollout_i = Thread(name='rollout' + str(i), target=self.MC_rollout_par,
                                      args=(p_start_state, policy, expected_svf, lock_expected_svf))
                    rollouts.append(rollout_i)

                ##Start all threads
                for r in rollouts:
                    r.start()

                ##Wait for all threads to finish
                for r in rollouts:
                    r.join()
            else: ##Option not to parallel
                rollouts.append( self.MC_rollout(p_start_state, policy) )
               
        if not self.par: 
            for state_visitation_counts in rollouts:
                np.add(expected_svf, state_visitation_counts, expected_svf) ##Add in place, right?

        expected_svf[:, 1:] = expected_svf[:, 1:] / self.mc_rollouts
        return expected_svf.sum(axis=1), policy

    ##Start state frequencies based on trajectories - Count each start state and return frequencies.
    def find_start_state_freq(self):

        start_state_count = np.zeros(self.n_states)
        for trajectory in self.trajectories:
            start_state_count[trajectory[0, 0]] += 1
        p_start_state = start_state_count / self.trajectories.shape[0]

        return p_start_state

    ##Run MC rollout for rollout_horizon steps and record counts of visited states.
    def MC_rollout(self, p_start_state, policy):

        state_visitation_counts = np.zeros((self.n_states, self.rollout_horizon))

        ##Pick a start state
        state = np.random.choice(np.arange(self.n_states), p=p_start_state)

        for t in range(1, self.rollout_horizon):

            state_valid_actions = self.valid_actions[state]

            ##Reached terminal state (only one valid action - relevant for Pacman) - finish rollout
            if len(state_valid_actions) == 1: break

            ##Select a valid action from policy
            action = np.random.choice(state_valid_actions, p=policy[state][state_valid_actions]/sum(policy[state][state_valid_actions]))

            ##Execute action by selecting state from transition probabilities
            next_state = np.random.choice(np.arange(self.n_states), p=self.transition_probability[state, action, :])

            ##Update state visitation count
            state_visitation_counts[next_state, t] += 1

            state = next_state

        return state_visitation_counts

    ##Run MC rollout for rollout_horizon steps and record counts of visited states.
    def MC_rollout_par(self, p_start_state, policy, expected_svf, lock_expected_svf):

        state_visitation_counts = np.zeros((self.n_states, self.rollout_horizon))

        ##Pick a start state
        state = np.random.choice(np.arange(self.n_states), p=p_start_state)

        for t in range(1, self.rollout_horizon):

            state_valid_actions = self.valid_actions[state]

            ##Reached terminal state (only one valid action - relevant for Pacman) - finish rollout
            if len(state_valid_actions) == 1: break

            ##Select a valid action from policy
            action = np.random.choice(state_valid_actions, p=policy[state][state_valid_actions]/sum(policy[state][state_valid_actions]))

            ##Execute action by selecting state from transition probabilities
            next_state = np.random.choice(np.arange(self.n_states), p=self.transition_probability[state, action, :])

            ##Update state visitation count
            state_visitation_counts[next_state, t] += 1

            state = next_state

        ##Lock general expected_svf matrix
        lock_expected_svf.acquire()
        try:
            ##Add rollout counts to general matrix
            np.add(expected_svf, state_visitation_counts, expected_svf)
        finally:
            lock_expected_svf.release()
        return

    ##Direct calculation of state visitation frequencies (alg. 1 from paper) - no MC rollouts (original version)
    def find_expected_svf(self, r, p_start_state):

        ##Calculates probability for each action and state (lines 1-3 alg 1)
        policy = find_policy(self.n_states, self.n_actions, self.transition_probability, r,
                             self.discount, valid_actions=self.valid_actions, consider_valid_only=True)

        expected_svf = np.tile(p_start_state, (self.rollout_horizon, 1)).T
        for t in range(1, self.rollout_horizon):
            expected_svf[:, t] = 0
            for i, j, k in product(range(self.n_states), range(self.n_actions), range(self.n_states)): #line 5 alg 1
                expected_svf[k, t] += (expected_svf[i, t-1] *
                                      policy[i, j] * ##Stochastic policy
                                       self.transition_probability[i, j, k])

        return expected_svf.sum(axis=1), policy

    ##Wrapper function to calculate rewards and extract policy based on them
    def calculate_policy(self, n_actions, transition_probability, feature_matrix, trajectories, valid_actions={}, return_rewards=False):

        ##IRL rewards
        rewards, _, feature_rewards = self.irl(n_actions, transition_probability, feature_matrix, trajectories, valid_actions)

        ##Reconstruct policy based on learned rewards
        policy = find_policy(self.n_states, self.n_actions,self.transition_probability, rewards,
                    self.discount, stochastic=False, valid_actions=self.valid_actions, consider_valid_only=True)

        if return_rewards:
            return policy , feature_rewards
        else:
            return policy

    ##Debugging##
    def get_log_likelihood(self, feature_counts_by_trajectory, alpha):

        n_trajectories = feature_counts_by_trajectory.shape[0]

        ##Sum of feature counts multiplied by reward weights
        trajectory_reward = np.dot(feature_counts_by_trajectory, alpha)

        ##exponented trajectory reward
        exp_trajectory_reward = np.exp(trajectory_reward)

        log_likelihood = (sum(trajectory_reward) - np.log(sum(exp_trajectory_reward))) / n_trajectories

        return log_likelihood

    def plot_list(self, values_list, plot_name="", by="Iteration"):

        plt.title(plot_name + " by " + by)
        plt.plot(range(len(values_list)), values_list)
        plt.xlabel(by + " #")
        plt.ylabel(plot_name)

        plt.show()
        plt.clf()
        plt.close()

        