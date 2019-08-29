import numpy as np
from scipy.optimize import linprog
from copy import deepcopy
from random import shuffle

class MachineTeaching():

    def __init__(self, init_states, optimal_actions, sa_fcounts, num_actions, trajectory_length=1, summary_length=5, candidate_trajectories=[], candidate_trajectory_indices=[], constraint_set=None):

        self.init_states = init_states ##Indices of initial states
        self.optimal_actions = optimal_actions
        self.num_actions = num_actions
        self.sa_fcounts = sa_fcounts ##Dictionairy of state, action expected feature counts
        self.trajectory_length = trajectory_length ##Number of steps in each trajectory
        self.summary_length = summary_length ##Total num of states allowed in summary
        self.candidate_trajectories = candidate_trajectories ##All possible candidate trajectories to select from - trajectory: list of feature rep. states, list of actions
        self.candidate_trajectory_indices = candidate_trajectory_indices ##Indices of states for each trajectory
        self.constraint_set = constraint_set


    ##Main procedure
    def set_cover_optimal_teaching(self):
        ##Get non-redundant constraint set
        if self.constraint_set is None:
            self.constraint_set = self.get_constraint_set()

        ##Get candidate trajectories of length trajectory_length starting from each initial state
        candidate_trajectories = self.candidate_trajectories
        candidate_trajectory_indices = self.candidate_trajectory_indices

        ##Create boolean bookeeping to monitor which constraints were covered
        covered = [False]*len(self.constraint_set)

        ##For each candidate trajectory check how many uncovered constraints it covers and find one with max added covers
        ##Stop when all constraints are covered or summary_length budget reached
        total_covered = 0
        total_states = 0
        opt_trajectories = []
        opt_trajectory_indices = []
        while ((total_covered < len(self.constraint_set) and total_states + self.trajectory_length <= self.summary_length)): # or (all_constraints and total_covered < total_constraints):
            max_count = 0
            best_trajectory = None
            constraints_added = None

            indices = np.arange( len( candidate_trajectories ) )
            shuffle( indices )
            candidate_trajectories = list( np.array( candidate_trajectories )[ list( indices ) ] )
            candidate_trajectory_indices = list( np.array( candidate_trajectory_indices )[ list( indices ) ] )
            for i , trajectory in enumerate( candidate_trajectories ):
                ##Calculate halfspace constraints for states in trajectory
                trajectory_constraints = self.get_halfspace_constraints(candidate_trajectory_indices[i], trajectory[1])

                ##Count number of new constraints covered by trajectory
                count = self.count_new_covers(trajectory_constraints, self.constraint_set, covered)

                if (count > max_count):
                    max_count = count
                    best_trajectory = trajectory.copy()
                    best_trajectory_indices = candidate_trajectory_indices[ i ]
                    best_trajectory_idx = i
                    constraints_added = trajectory_constraints

            ##Get specified feature indices in trajectory
            for x in range(best_trajectory.shape[1]):
                best_trajectory[0][x] = best_trajectory[0][x]

            ##Add best trajectory to optimal list , update covered flags and increment total covered count
            opt_trajectories.append(best_trajectory)
            opt_trajectory_indices.append( best_trajectory_indices )
            covered = self.update_covered_constraints(constraints_added, self.constraint_set, covered)
            total_covered += max_count
            total_states += self.trajectory_length

            ##Remove best trajectory from candidate lists
            candidate_trajectories.pop(best_trajectory_idx)
            candidate_trajectory_indices.pop(best_trajectory_idx)

        ##If summary size is smaller than requested try adding random trajectories
        if total_states + self.trajectory_length <= self.summary_length: # and not all_constraints:
            n_add = int((self.summary_length - total_states)/self.trajectory_length)
            if n_add > 0:
                rand_idx = np.random.choice(len(candidate_trajectory_indices), n_add, replace=False)

                ##Add trajectories to list
                for idx in rand_idx:
                    add_trajectory = candidate_trajectories[idx].copy()
                    add_trajectory_indices = candidate_trajectory_indices[idx]
                    ##Get specified feature indices in trajectory
                    for x in range(add_trajectory.shape[1]):
                        add_trajectory[0][x] = add_trajectory[0][x]
                    opt_trajectories.append(add_trajectory)
                    opt_trajectory_indices.append(add_trajectory_indices)

        return opt_trajectories , opt_trajectory_indices

    ##Get halfspace constraints for given states and optimal actions
    def get_halfspace_constraints(self, state_indices, optimal_actions):

        constraints = []
        for i, state in enumerate(state_indices):

            opt_action = optimal_actions[i]
            mu_sa = self.sa_fcounts[(state, opt_action)]

            ##For each non optimal action, create a constraint: mu_sb - mu_sa <= 0 (should be mu_sa-mu_sb >= 0 but multiplied by -1 for LP problem solving)
            for action in range(self.num_actions):
                if action != opt_action:
                    mu_sb = self.sa_fcounts[(state, action)]

                    ##Add constraint to list
                    constraints.append(mu_sb - mu_sa)

        ##Normalize constraints and remove duplicates
        constraints = self.normalize(constraints)

        return constraints

    ##Normalize constraints - divide by L2 norm and remove duplicate constraints
    def normalize(self, constraints):

        ##If there is only one feature don't normalize (or else all constarints will be equal to 1 or -1)
        if constraints[0].shape[0] == 1:
            normalized_constraints = deepcopy(constraints)
        else:
            normalized_constraints = []
            for c in constraints:
                l2_norm = np.linalg.norm(c)
                if l2_norm == 0:
                    normalized_constraints.append(c)
                else:
                    normalized_constraints.append(c / l2_norm)

        ##Remove duplicate constraints
        #TODO: maybe change to np.allclose to check for equality with tolerance like in brown implementation
        unique_constraints = np.unique(np.array(normalized_constraints), axis=0)

        ##Remove trivial all zero constraints
        unique_constraints = unique_constraints[~np.all(unique_constraints == 0.0, axis=1)]
        return list(unique_constraints)

    ##Remove redundant halfspaces using linear programming
    def remove_redundant_halfspaces(self, constraints):

        lp_constraints = constraints
        non_redundant = deepcopy(lp_constraints)
        lp_status_dict = {0:0, 1:0, 2:0, 3:0}

        ##Solve LP problem with each constraint as objective and all other constraints as problem constraints:
        ##Maximize: c^T * x, subjecto to: A_ub * x <= b_ub --> (mu_sb - mu_sa)*x <= 0
        b = np.zeros(len(non_redundant) - 1)
        for obj_constraint in lp_constraints:

            ##Set objective constrains
            ##linprog solves only minimum problems so use max(f(x)) == -min(-f(x)) --> multiply constraint by -1
            c = obj_constraint*(-1)

            ##Set all other constraints
            A = [const for const in non_redundant if  not np.array_equal(const, obj_constraint)]

            ##Set variables to be unbounded
            bounds = (None, None)

            ##Solve LP problem
            lp_result = linprog(c, A_ub=A, b_ub=b, bounds=(bounds))

            ##Status of LP:
            ## 0 : Optimization terminated successfully
            ## 1 : Iteration limit reached
            ## 2 : Problem appears to be infeasible
            ## 3 : Problem appears to be unbounded
            lp_status = lp_result.status
            opt_value = 0.0
            lp_status_dict[lp_status] += 1

            if lp_status == 0:
                ##Get negative of optimal value (because of maximum formulation)
                opt_value = -lp_result.fun

            ##If optimal solution <= 0 constraint is redundant - remove from list
            ##Also remove if infeasible or iteration limit reached (keep if unbounded)
            if (lp_status == 0 and opt_value <= 0.0) or (lp_status == 1) or (lp_status == 2):
                non_redundant = A
                b = np.zeros(len(non_redundant) - 1)

                if b.size == 0: break ##Last constraint - finish

        #print("LP status: ", lp_status_dict)
        #print("Num constraints: ", len(non_redundant))
        return non_redundant


    ##Count how many uncovered constraints are covered by trajectory
    def count_new_covers(self, trajectory_constraints, constraint_set, covered):
        count = 0
        for traj_constraint in trajectory_constraints:
            for i, set_constraint in enumerate(constraint_set):
                ##Set constraint is uncovered - check if trajectory constraint covers it (with small tolerance)
                if not covered[i]:
                    if np.allclose(traj_constraint, set_constraint, rtol=0.0, atol=1e-04):
                        count += 1

        return count

    ##Update covered flags for new covered constraints
    def update_covered_constraints(self, constraints_added, constraint_set, covered):
        for new_constraint in constraints_added:
            for i, set_constraint in enumerate(constraint_set):
                if not covered[i]:
                    if np.allclose(new_constraint, set_constraint, rtol=0.0, atol=1e-04):
                        covered[i] = True

        return covered

    def get_constraint_set(self):
        ##Calculate halfspace constraints formed by all optimal actions in initial states
        constraints = self.get_halfspace_constraints(self.init_states, self.optimal_actions)

        ##Remove redundant halfspace constraints
        if len(constraints) > 1:
            constraint_set = self.remove_redundant_halfspaces(constraints)
        else:
            constraint_set = constraints

        return constraint_set



















