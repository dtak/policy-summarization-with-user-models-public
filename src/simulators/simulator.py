import os
import pickle
import numpy as np
import pylab as plt

from abc import ABCMeta , abstractmethod
from sklearn.cluster import KMeans
from time import time

from sklearn.preprocessing import StandardScaler

##Don't need to use ABC stuff, could then implement a lot of the standard data processing here!!
##Pass in **kw args, super( ) -- calls method in the super class -- see: https://github.com/dtak/expressive_nn_priors/blob/master/expressive_nn_priors/optimizers.py

class Simulator( object ):
	__metaclass__ = ABCMeta

	def __init__( self ):
		#self.value = value
		
		#super().__init__()

		self.trajectory_lengths = [ 1 , 2 , 3 , 4 , 5 ]
		self.trajectories_set = {}
		for trajectory_length in self.trajectory_lengths:
			self.trajectories_set[ trajectory_length ] = self.get_candidate_trajectories( trajectory_length )
		self.unique_state_features , self.unique_state_actions , self.unique_state_identifiers , self.trajectories_set_indices_into_unique_states = self.get_matrices( self.trajectories_set )
		self.unique_state_il_features = self.unique_state_features[ : , self.il_indices ]
		self.unique_state_irl_features = self.unique_state_features[ : , self.irl_indices ]
		self.standardized_unique_state_features , self.scaler = self.standardize_state_features( self.unique_state_features )
		self.standardized_unique_state_il_features = self.standardized_unique_state_features[ : , self.il_indices ]
		self.standardized_unique_state_irl_features = self.standardized_unique_state_features[ : , self.irl_indices ]

	@abstractmethod
	def get_state_action_fcounts( self ):
		pass

	@abstractmethod
	def get_candidate_trajectories( self ):
		pass

	@abstractmethod
	def get_policy_value( self ):
		pass

	def get_matrices( self , trajectories_set ):

		##Convert trajectories to matrices
		unique_state_features = []
		unique_state_actions = []
		unique_state_identifiers = [] ##Unique ids of the states
		trajectories_set_indices = {}
		for trajectory_length in self.trajectory_lengths:
			candidate_trajectories = trajectories_set[ trajectory_length ]

			trajectories_set_indices[ trajectory_length ] = []
			for i , trajectory in enumerate( candidate_trajectories ):
				trajectories_set_indices[ trajectory_length ].append( [] )

				for j in range( len( trajectory[ 0 ] ) ): 
					state_features = trajectory[ 0 ][ j ]
					state_action = trajectory[ 1 ][ j ]
					state_identifier = trajectory[ 2 ][ j ]

					##Add to unique list only if not already there!
					if state_identifier not in unique_state_identifiers:
						unique_state_identifiers.append( state_identifier )
						unique_state_features.append( state_features )
						unique_state_actions.append( state_action )

					trajectories_set_indices[ trajectory_length ][ i ].append( unique_state_identifiers.index( state_identifier ) )
			trajectories_set_indices[ trajectory_length ] = np.array( trajectories_set_indices[ trajectory_length ] )

		return np.array( unique_state_features ) , np.array( unique_state_actions ) , np.array( unique_state_identifiers ) , trajectories_set_indices

	def standardize_state_features( self , X , unlabeled_indices=[] , test_indices=[] ):

		standardized_X = np.copy( X )

		test_indices = list( unlabeled_indices ) + list( test_indices )
		train_indices = list( np.setdiff1d( np.arange( X.shape[ 0 ] ) , test_indices ) )

		scaler = StandardScaler()
		standardized_X[ train_indices ] = scaler.fit_transform( X[ train_indices , : ] )

		if len( test_indices ):
			standardized_X[ test_indices ] = scaler.transform( X[ test_indices , : ] )

		return standardized_X, scaler

	def convert_trajectories_to_sa_tuples(self, trajectories, all_actions):

		##Convert trajectories of state indices to (state, action) pairs - for irl reconstruction
		sa_trajectories = []
		for trajectory in trajectories:
			sa_trajectory = []

			for state in trajectory:
				action = all_actions[state]
				sa_trajectory.append((state, action))

			sa_trajectories.append(sa_trajectory)

		return np.array(sa_trajectories)

	def map_discretized_policy_to_original_states(self, policy):

		##By default fits discrete domains and does nothing, overriden in HIV simulator
		return policy

	def map_reconstructed_policy_to_discretized_policy(self, original_policy):

		##By default fits discrete domains and does nothing, overriden in HIV simulator
		return None



