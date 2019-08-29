##External imports
import os
import sys
import pickle
import subprocess
import numpy as np
import pylab as plt
from time import time
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid

##My utils
import utils

from models.grf import GRF
from models.irl_maxent import Maxent
from extraction_methods.active_learning import ActiveLearning
from extraction_methods.machine_teaching import MachineTeaching

class HyperparameterComparisonPipeline:

	def __init__( self , accuracy_threshold , summary_length , extract_hyperparams , reconstruct_hyperparams , simulator ):

		self.accuracy_threshold = accuracy_threshold
		self.summary_length = summary_length
		self.simulator = simulator

		##Cache param stuff
		self.update_parameters( summary_length , extract_hyperparams , reconstruct_hyperparams )

		##Cache quanitites for machine teaching
		self.cache_mt_intermediate_quantities()

	##These quantities are calculated once per pipeline
	def cache_mt_intermediate_quantities( self ):

		##State action feature counts
		self.sa_fcounts = self.simulator.get_state_action_fcounts()

		##Create temporary machine teaching object for constraint calculation, later we create a new one for extraction
		unique_state_actions = list( self.simulator.unique_state_actions )
		unique_state_numbers = list( np.arange( len( unique_state_actions ) ) )
		mt = MachineTeaching( unique_state_numbers , unique_state_actions , self.sa_fcounts , len( self.simulator.action_names ) )
		self.mt_constraint_set = mt.get_constraint_set()

	##Useful for re-using the same object with the data already extracted
	def update_parameters( self , summary_length , extract_hyperparams , reconstruct_hyperparams ): 
		self.summary_length = summary_length

		self.extract_hyperparams_list = [ params for params in ParameterGrid( extract_hyperparams ) ]
		self.reconstruct_hyperparams_list = [ params for params in ParameterGrid( reconstruct_hyperparams ) ]
		self.extract_names = [ utils.get_model_name( hyperparams ) for hyperparams in self.extract_hyperparams_list ]
		self.reconstruct_names = [ utils.get_model_name( hyperparams ) for hyperparams in self.reconstruct_hyperparams_list ]

	def instantiate_model_from_hyperparams( self , hyperparams ):
		if hyperparams[ "model" ] == "grf":
			model = GRF( hyperparams[ "kernel_params" ] , n_classes=self.simulator.n_actions )
			if "algorithm" in hyperparams: ##It's the extraction step
				if hyperparams[ "algorithm" ] == "al":
					model = ActiveLearning( model=model , n_queries=self.summary_length )
				elif hyperparams[ "algorithm" ] == "random":
					model = ActiveLearning( model=model , n_queries=self.summary_length , strategy="random" ) ##Random extraction!!
				else:
					assert False , str( hyperparams )
		elif hyperparams[ "model" ] == "irl":
			if "maxent_params" in hyperparams:  ##IRL Reconstruction
				model = Maxent( **hyperparams[ "maxent_params" ] )
			else: ##During MT extraction
				model = None
		else:
			assert False , str( hyperparams[ "model" ] )
		return model

	##This is the main inner loop
	##It computes the reconstruction accuracy and policy value 
	##for each extraction and reconstruction model pair
	##It also saves the extracted summaries and the reconstructions
	def compute_extraction_by_reconstruction_grids( self ):

		self.summaries_list = []
		self.reconstructions_grid = []
		self.accuracy_grid = []
		self.value_grid = []

		for i , extract_hyperparams in enumerate( self.extract_hyperparams_list ):

			if extract_hyperparams[ "model" ] == "grf":
				summary_of_indices_into_unique_states , labeled_indices_into_unique_states , unlabeled_indices_into_unique_states = self.extract_summary( extract_hyperparams )
			else:
				assert extract_hyperparams[ "model" ] == "irl" , "Model must be grf or irl! Not: " + str( extract_hyperparams[ "model" ] )
				summary_of_indices_into_unique_states , labeled_indices_into_unique_states , unlabeled_indices_into_unique_states = self.extract_summary( extract_hyperparams )				
			
			self.summaries_list.append( summary_of_indices_into_unique_states )

			reconstructions = []
			accuracies = []
			values = []
			for j , reconstruct_hyperparams in enumerate( self.reconstruct_hyperparams_list ):
				reconstructed_unique_state_actions, discretized_policy = self.compute_reconstruction( reconstruct_hyperparams , summary_of_indices_into_unique_states , labeled_indices_into_unique_states , unlabeled_indices_into_unique_states )

				##Compute accuracy
				accuracy = accuracy_score( self.simulator.unique_state_actions[ unlabeled_indices_into_unique_states ] , reconstructed_unique_state_actions[ unlabeled_indices_into_unique_states ] )
				print( accuracy )

				##Need discretized policy for value calculation in HIV
				if discretized_policy is None:
					reconstructed_policy = reconstructed_unique_state_actions
				else:
					reconstructed_policy = discretized_policy
				##Compute value difference between original and reconstructed policy
				reconstruction_value = self.simulator.get_policy_value( policy=reconstructed_policy , n_simulations=self.simulator.n_value_simulations , n_steps=self.simulator.n_value_steps )
				value_difference = abs( self.simulator.policy_value - reconstruction_value )
				
				accuracies.append( accuracy )
				values.append( value_difference )
				reconstructions.append(reconstructed_unique_state_actions)

			self.reconstructions_grid.append( reconstructions )
			self.accuracy_grid.append( accuracies )
			self.value_grid.append( values )
		
		self.accuracy_grid = np.array( self.accuracy_grid )
		self.value_grid = np.array( self.value_grid )

		print( "\nResults")
		print( "Accuracy: " + str( self.accuracy_grid ) )
		print( "Value: " + str( self.value_grid ) )
		print( "---\n\n" )

	def extract_summary( self , extract_hyperparams ):
		extraction_model = self.instantiate_model_from_hyperparams( extract_hyperparams )

		if extract_hyperparams[ "algorithm" ] == "random" or extract_hyperparams[ "model" ] == "grf":
			if "trajectry_length" in extract_hyperparams:
				assert extract_hyperparams[ "algorithm" ] == "random"
				extraction_model.fit( self.simulator.standardized_unique_state_il_features , self.simulator.unique_state_actions , self.simulator.trajectories_set_indices_into_unique_states[ extract_hyperparams[ "trajectory_length" ] ] )	
			else:
				extraction_model.fit( self.simulator.standardized_unique_state_il_features , self.simulator.unique_state_actions , self.simulator.trajectories_set_indices_into_unique_states[ 1 ] )	
			summary_of_indices_into_unique_states = self.simulator.trajectories_set_indices_into_unique_states[ 1 ][ extraction_model.queried_trajectory_indices ]
			labeled_indices_into_unique_states = np.unique( summary_of_indices_into_unique_states ) ##Points in X in the summary
			unlabeled_indices_into_unique_states = np.setdiff1d( np.arange( self.simulator.standardized_unique_state_il_features.shape[ 0 ] ) , labeled_indices_into_unique_states ) ##Points in X not in the summary, 1 per unique coordinates id
			
		elif extract_hyperparams[ "model" ] == "irl":
			machine_teaching = MachineTeaching( np.arange( len( self.simulator.unique_state_identifiers ) ) , self.simulator.unique_state_actions , self.sa_fcounts , 
												num_actions=self.simulator.n_actions, trajectory_length=extract_hyperparams[ "trajectory_length" ], summary_length=self.summary_length ,
												candidate_trajectories=self.simulator.trajectories_set[ extract_hyperparams[ "trajectory_length" ] ] , 
												candidate_trajectory_indices=self.simulator.trajectories_set_indices_into_unique_states[ extract_hyperparams[ "trajectory_length" ] ] , 
												constraint_set=self.mt_constraint_set)
			_ , summary_of_indices_into_unique_states = machine_teaching.set_cover_optimal_teaching()
			labeled_indices_into_unique_states = np.unique( summary_of_indices_into_unique_states ) ##Points in X in the summary
			unlabeled_indices_into_unique_states = np.setdiff1d( np.arange( self.simulator.standardized_unique_state_il_features.shape[ 0 ] ) , labeled_indices_into_unique_states ) ##Points in X not in the summary, 1 per unique coordinates id

		else:
			assert False , "Only grf, irl, extraction implemented, not " + str( extract_hyperparams[ "model" ] )
		
		return summary_of_indices_into_unique_states , labeled_indices_into_unique_states , unlabeled_indices_into_unique_states

	def compute_reconstruction( self , reconstruct_hyperparams , summary_of_indices_into_unique_states , labeled_indices_into_unique_states , unlabeled_indices_into_unique_states ):
		reconstruction_model = self.instantiate_model_from_hyperparams( reconstruct_hyperparams )

		if reconstruct_hyperparams[ "model" ] == "grf":
			print( self.simulator.standardized_unique_state_il_features.shape )
			predictions_of_unique_state_actions_for_unlabeled = reconstruction_model.predict( self.simulator.standardized_unique_state_il_features , self.simulator.unique_state_actions , unlabeled_indices_into_unique_states )
		elif reconstruct_hyperparams[ "model" ] == "irl":

			##Convert trajectories to state-action tuples
			irl_trajectories = self.simulator.convert_trajectories_to_sa_tuples(summary_of_indices_into_unique_states, self.simulator.unique_state_actions)

			if self.simulator.domain == "hiv":
				irl_policy = reconstruction_model.calculate_policy(self.simulator.n_actions, self.simulator.transition_probas, self.simulator.features, irl_trajectories , self.simulator.valid_actions)
			else:
				irl_policy = reconstruction_model.calculate_policy(self.simulator.n_actions, self.simulator.transition_probas, self.simulator.unique_state_irl_features, irl_trajectories , self.simulator.valid_actions)

			##Map IRL policy to original states (in discrete domains does nothing, in continuous uses clusters mapping to map dicretized policy back to original states)
			original_states_policy = self.simulator.map_discretized_policy_to_original_states(irl_policy)

			##Get predictions only for the unlabeled points
			predictions_of_unique_state_actions_for_unlabeled = original_states_policy[ unlabeled_indices_into_unique_states ]

		else:
			assert False , "Only grf, irl, extraction implemented, not " + str( reconstruct_hyperparams[ "model" ] )

		reconstructed_policy = self.simulator.unique_state_actions.copy() ##Use seen actions for states you've seen
		reconstructed_policy[ unlabeled_indices_into_unique_states ] = predictions_of_unique_state_actions_for_unlabeled

		discretized_policy = self.simulator.map_reconstructed_policy_to_discretized_policy(reconstructed_policy)

		return reconstructed_policy, discretized_policy



