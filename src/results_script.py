import os
import time
import pickle
import argparse
import numpy as np
import scipy.stats

import utils

from simulators.simulator import Simulator
#from simulators.hiv_simulator import HIVSimulator
#from simulators.pacman_simulator import PacmanSimulator
from simulators.random_gridworld_simulator import RandomGridworldSimulator

from sklearn.model_selection import ParameterGrid
from hyperparameter_comparison_pipeline import HyperparameterComparisonPipeline


##Get the initialized simulator for a domain. Set parameters here!
def get_simulator( domain ):

	if domain == "gridworld":
		simulator = RandomGridworldSimulator( max_x=9 , max_y=9 , discount=0.95 , n_features=5 , random=False , n_value_simulations=1, n_value_steps=10 )
	elif domain == "hivsimulator":
		simulator = HIVSimulator( perturb_rate=0.05 , discount=0.98 , ins=20 , eps=0. , n_episodes=5 , episode_length=200 ,
									n_clusters=100 , rollout_horizon=25 , n_value_simulations=5, n_value_steps=200 )
	elif domain == "pacman":
		simulator = PacmanSimulator( agent="towards_food" , grid="smallGrid" , ghost_agent="DirectionalGhost" , n_value_simulations=1, n_value_steps=10  )
	else:
		assert False , str( domain ) + " not implemented yet!"
	return simulator

##Get the lists of possible parameters here.  Set parameters here!
def get_params( domain , cross_pairs , debug ):

	if not cross_pairs: ##Hyperparameter search!
		if domain == "gridworld": 
			domain_discount = get_simulator( domain ).discount
			domain_rollout = get_simulator( domain ).rollout_horizon
			learning_rate = 1.
			gammas = [ 0.001 , 0.01 , 0.1 , 1. ]
		elif domain == "hivsimulator":
			domain_discount = 0.98
			domain_rollout = 25
			learning_rate = 0.01
			gammas = [ 0.01 , 0.1 , 1. , 10. ]
		else:
			assert domain == "pacman" , str( domain ) + " not valid!"
			domain_discount = 0.95
			domain_rollout = 10 
			learning_rate = 0.1
			gammas = [ 0.001 , 0.01 , 0.1 , 1. ]

		if debug: ##A short run to check things!!

			grf_params = ParameterGrid( [ { "kernel": [ "rbf" ] , "gamma": [ 1. ] } ] )
			maxent_params = ParameterGrid( [ { "discount": [ domain_discount ] , "epochs": [ 10 ] , "learning_rate": [ learning_rate ] , 
												"gd_threshold": [ 1e-5 ] , "mc_rollouts": [ 0 ] , "rollout_horizon": [ domain_rollout ] ,  ##10 rollout!!
												"par": [ False ] } ] )

			##Set options for hyperparameter05
			extract_hyperparams = [ { "algorithm": [ "al" ] , "model": [ "grf" ] , "kernel_params": grf_params } ,
										{ "algorithm": [ "random" ] , "trajectory_length": [ 1 ] , "model": [ "grf" ] , "kernel_params": [ { "kernel": "rbf" } ] } ,
										{ "algorithm": [ "mt" ] , "model": [ "irl" ] , "trajectory_length": [ 3 ] , "match_rollout": [ False ] }
									]
			reconstruct_hyperparams = [ { "model": [ "grf" ] , "kernel_params": grf_params } ,
										{ "model": [ "irl" ] , "maxent_params": maxent_params , "match_rollout": [ False ]  } 
									]
			summary_sizes = [ 12 , 60 ] ##Factors of 12 so all summary lengths complete

		else:

			discounts = [ domain_discount ]
			degrees = [ 1 , 2 , 3  ]
			trajectory_lengths = [ 1 , 2 , 3 , 4 ]

			grf_params = ParameterGrid( [ { "kernel": [ "rbf" ] , "gamma": gammas } , { "kernel": [ "poly" ] , "gamma": gammas , "degree": degrees } ] )
			maxent_params = ParameterGrid( [ { "discount": discounts , "epochs": [ 100 ] , "learning_rate": [ learning_rate ] , 
												"gd_threshold": [ 1e-5 ] , "mc_rollouts": [ 0 ] , "rollout_horizon": [ domain_rollout ] ,  ##10 rollout!!
												"par": [ False ] } ] )

			##Set options for hyperparameter05
			extract_hyperparams = [ { "algorithm": [ "al" ] , "model": [ "grf" ] , "kernel_params": grf_params } ,
										{ "algorithm": [ "random" ] , "trajectory_length": trajectory_lengths , "model": [ "grf" ] , "kernel_params": [ { "kernel": "rbf" } ] } ,
										{ "algorithm": [ "mt" ] , "model": [ "irl" ] , "trajectory_length": trajectory_lengths , "match_rollout": [ False ] }
									]
			reconstruct_hyperparams = [ { "model": [ "grf" ] , "kernel_params": grf_params } ,
										{ "model": [ "irl" ] , "maxent_params": maxent_params , "match_rollout": [ False ]  } 
									]
				
			summary_sizes = [ 12 , 24 , 36 , 48 , 60 ] ##Factors of 12 so all summary lengths complete

	else: ##Best hyperparameters
		if domain == "gridworld":
			domain_discount = get_simulator( domain ).discount
			domain_rollout = get_simulator( domain ).rollout_horizon
			learning_rate = 1.

			discounts = [ domain_discount ]

			grf_params = ParameterGrid( [ { "kernel": [ "poly" ] , "gamma": [ 0.1 ] , "degree": [ 2 , 3 ] } ] )
			maxent_params = ParameterGrid( [ { "discount": discounts , "epochs": [ 100 ] , "learning_rate": [ learning_rate ] ,
								"gd_threshold": [ 1e-5 ] , "mc_rollouts": [ 0 ] , "rollout_horizon": [ domain_rollout ] ,
								"par": [ False ] } ] )

			##Set options for hyperparameter05
			extract_hyperparams = [ { "algorithm": [ "al" ] , "model": [ "grf" ] , "kernel_params": grf_params } ,
						{ "algorithm": [ "mt" ] , "model": [ "irl" ] , "trajectory_length": [ 3 , 4 ] , "match_rollout": [ False ] }
						]
			reconstruct_hyperparams = [ { "model": [ "grf" ] , "kernel_params": grf_params } ,
							{ "model": [ "irl" ] , "maxent_params": maxent_params , "match_rollout": [ False ]  }
						]

			summary_sizes = [ 24 ]

		elif domain == "hivsimulator":
			domain_discount = 0.98
			domain_rollout = 25
			learning_rate = 0.01

			discounts = [ domain_discount ]

			grf_params = ParameterGrid( [ { "kernel": [ "rbf" ] , "gamma": [ 1. ] } ] )
			maxent_params = ParameterGrid( [ { "discount": discounts , "epochs": [ 100 ] , "learning_rate": [ learning_rate ] ,
								"gd_threshold": [ 1e-5 ] , "mc_rollouts": [ 0 ] , "rollout_horizon": [ domain_rollout ] ,
								"par": [ False ] } ] )

			##Set options for hyperparameter05
			extract_hyperparams = [ { "algorithm": [ "al" ] , "model": [ "grf" ] , "kernel_params": grf_params } ,
						{ "algorithm": [ "mt" ] , "model": [ "irl" ] , "trajectory_length": [ 1 , 2 , 3 , 4 ] , "match_rollout": [ False ] }
						]
			reconstruct_hyperparams = [ { "model": [ "grf" ] , "kernel_params": grf_params } ,
							{ "model": [ "irl" ] , "maxent_params": maxent_params , "match_rollout": [ False ]  }
						]

			summary_sizes = [ 24 ]

		elif domain == "pacman":
			domain_discount = 0.95
			domain_rollout = 10 
			learning_rate = 0.1
			discounts = [ domain_discount ]

			grf_params = ParameterGrid( [ { "kernel": [ "rbf" ] , "gamma": [ 1. ] } ] )
			maxent_params = ParameterGrid( [ { "discount": discounts , "epochs": [ 100 ] , "learning_rate": [ learning_rate ] ,
								"gd_threshold": [ 1e-5 ] , "mc_rollouts": [ 0 ] , "rollout_horizon": [ domain_rollout ] ,
								"par": [ False ] } ] )

			##Set options for hyperparameter05
			extract_hyperparams = [ { "algorithm": [ "al" ] , "model": [ "grf" ] , "kernel_params": grf_params } ,
						{ "algorithm": [ "mt" ] , "model": [ "irl" ] , "trajectory_length": [ 1 , 2 , 3 , 4 ] , "match_rollout": [ False ] }
						]
			reconstruct_hyperparams = [ { "model": [ "grf" ] , "kernel_params": grf_params } ,
							{ "model": [ "irl" ] , "maxent_params": maxent_params , "match_rollout": [ False ]  }
						]

			summary_sizes = [ 12 ]

		else:

			assert False , "Hyperparam sweep not yet done!"

	return extract_hyperparams, reconstruct_hyperparams , summary_sizes

##Initialize all the lists of extractions and the corresponding exmpty lists for saving data!
def initialize_lists( extract_hyperparams , reconstruct_hyperparams , cross_pairs ):
	##Make lists of hyperparams for use in model later
	extract_hyperparams_list = [ params for params in ParameterGrid( extract_hyperparams ) ]
	reconstruct_hyperparams_list = [ params for params in ParameterGrid( reconstruct_hyperparams ) ]

	##Initialize dictionary with keys
	lists = [ "extract_names" , "extract_list" , "reconstruct_names" , "reconstruct_list" , "summary_sizes_list" , 
				"accuracies_mean" , "accuracies_std_dev" , "accuracies_95_ci" , "accuracies_raw" ,
				"values_mean" , "values_std_dev" , "values_95_ci" , "values_raw" ]
	data = { list_name : [] for list_name in lists }

	##Put together arrays to save data for each run
	for extract_hyperparams_choice in extract_hyperparams_list:
		if not cross_pairs: ##If hyperparam search, 
			if extract_hyperparams_choice[ "algorithm" ] == "random": ##Try all reconstructions on random summaries!!
				reconstruct_hyperparams_choices = reconstruct_hyperparams_list
			else:
				reconstruct_hyperparams_choices = [ utils.find_corresponding_reconstruction( extract_hyperparams_choice , reconstruct_hyperparams_list ) ]
		else: ##If full table, all combinations
			reconstruct_hyperparams_choices = reconstruct_hyperparams_list

		for reconstruct_hyperparams_choice in reconstruct_hyperparams_choices:
			data[ "extract_names" ].append( utils.get_model_name( extract_hyperparams_choice ) )
			data[ "reconstruct_names" ].append( utils.get_model_name( reconstruct_hyperparams_choice ) )
			data[ "extract_list" ].append( { key : [ value ] for key , value in extract_hyperparams_choice.items() } )
			data[ "reconstruct_list" ].append( { key: [ value ] for key , value in reconstruct_hyperparams_choice.items() }	 )
			for list_name in lists[ 4: ]:
				data[ list_name ].append( [] )

	return data

def get_simulator_and_pipeline( params ):
	domain , restart , extract_hyperparams , reconstruct_hyperparams , n_extraction_restarts , n_reconstruction_restarts , load_from_file = params
		
	fname = "../policy_data/"+domain+"_restart_"+str(restart)+".p"
	if load_from_file and os.path.isfile( fname ):
		simulator , pipeline = pickle.load( open( fname , "rb" ) )
	else:

		simulator = get_simulator( domain )
		##Note, the code updates true summary length and params later!!  We just want to cache MT stuff now.
		pipeline = HyperparameterComparisonPipeline( accuracy_threshold=0. , summary_length=1 ,
					extract_hyperparams=extract_hyperparams , reconstruct_hyperparams=reconstruct_hyperparams , simulator=simulator )

		if load_from_file:
			pickle.dump( ( simulator , pipeline ) , open( fname , "wb" ) )

	return simulator , pipeline

def get_simulators_for_each_restart( domain , n_restarts , extract_hyperparams , reconstruct_hyperparams ,
									 n_extraction_restarts , n_reconstruction_restarts , load_from_file ):
	params_list = []
	for restart in range( n_restarts ):
		params_list.append( ( domain , restart , extract_hyperparams , reconstruct_hyperparams , 
							n_extraction_restarts , n_reconstruction_restarts , load_from_file ) )

	run_outputs = []
	for params in params_list:
		run_outputs.append( get_simulator_and_pipeline( params ) )

	simulators = [] 
	pipelines = []
	for simulator , pipeline in run_outputs:
		##Save simulators and corresponing pipelines for later use!
		simulators.append( simulator )
		pipelines.append( pipeline )

	return simulators , pipelines

#extract_hyperparams , reconstruct_hyperparams , 
def compute_data_for_all_hyperparameter_runs( summary_sizes , data , pipelines , simulators , n_restarts ):

	runs = []
	for i , summary_length in enumerate( summary_sizes ):
		for j in range( len( data[ "extract_list" ] ) ):
			extract_hyperparams = data[ "extract_list" ][ j ]
			reconstruct_hyperparams = data[ "reconstruct_list" ][ j ]

			##Don't run for trajectory sizes that don't use the full summary size allowance
			if extract_hyperparams[ "model" ][ 0 ] != "irl" or summary_length % extract_hyperparams[ "trajectory_length" ][ 0 ] == 0:
				for k in range( n_restarts ):
					runs.append( ( summary_length , extract_hyperparams , reconstruct_hyperparams , pipelines[ k ] , simulators[ k ] ) )

	print( str( len( runs ) ) + " runs" )
	return runs

def compute_accuracy_and_value_for_parameter_setting( run_data ):

	summary_length , extract_hyperparams , reconstruct_hyperparams , pipeline , simulator = run_data

	print( extract_hyperparams )

	pipeline.update_parameters( summary_length , extract_hyperparams , reconstruct_hyperparams ) ##Update with new params!!
	pipeline.compute_extraction_by_reconstruction_grids()

	accuracy = pipeline.accuracy_grid[ 0 ][ 0 ]
	value = pipeline.value_grid[ 0 ][ 0 ]

	assert pipeline.accuracy_grid.shape == ( 1 , 1 ) , str( pipeline.accuracy_grid )
	assert pipeline.value_grid.shape == ( 1 , 1 ) , str( pipeline.value_grid )

	return ( summary_length , extract_hyperparams , reconstruct_hyperparams , pipeline , simulator , accuracy , value )

##Computes the accuracy for each parameters setting for each 
def compute_accuracy_and_value_for_each_parameter_setting_parallel( summary_sizes , data , simulators , pipelines , runs ):
	run_outputs = []
	for i , run_data in enumerate( runs ):
		print( "Run " + str( i ) + " out of " + str( len( runs ) ) )
		run_outputs.append( compute_accuracy_and_value_for_parameter_setting( run_data ) )
	return run_outputs

def build_data_dict_from_run_outputs( summary_sizes , data , run_outputs ):
	##For every option, get the run outputs corresponding to it!
	for i , summary_length in enumerate( summary_sizes ):
		for j in range( len( data[ "extract_list" ] ) ):
			extract_hyperparams = data[ "extract_list" ][ j ]
			reconstruct_hyperparams = data[ "reconstruct_list" ][ j ]
			##Don't run for trajectory sizes that don't use the full summary size allowance
			if extract_hyperparams[ "model" ][ 0 ] != "irl" or summary_length % extract_hyperparams[ "trajectory_length" ][ 0 ] == 0:
				accuracies = []
				values = []
				for run_output in run_outputs:
					run_summary_length , run_extract_hyperparams , run_reconstruct_hyperparams , pipeline , simulator , accuracy , value = run_output
					##If the run output corresponds to the current hyperparameter settings
					if run_summary_length == summary_length and run_extract_hyperparams == extract_hyperparams and run_reconstruct_hyperparams == reconstruct_hyperparams:	
						##Save the corresponding accuracies for this restart!
						accuracies.append( accuracy )
						values.append( value )
						print( accuracy )

				##Save average accuracies and standard errors in the data dictionary
				data[ "accuracies_mean" ][ j ].append( np.mean( accuracies ) )
				data[ "accuracies_std_dev" ][ j ].append( np.std( accuracies ) )
				data[ "accuracies_95_ci" ][ j ].append( 1.96 * scipy.stats.sem( accuracies ) )
				data[ "accuracies_raw" ][ j ].append( accuracies )

				##Values
				data[ "values_mean" ][ j ].append( np.mean( values ) )
				data[ "values_std_dev" ][ j ].append( np.std( values ) )
				data[ "values_95_ci" ][ j ].append( 1.96 * scipy.stats.sem( values ) )
				data[ "values_raw" ][ j ].append( values )

				##Summary sizes
				data[ "summary_sizes_list" ][ j ].append( summary_length )

	return data

def run_results( domain , cross_pairs , restart_number=-1 , n_restarts=1 , n_extraction_restarts=1 ,
								   n_reconstruction_restarts=1 , load_from_file=False , debug=False ):

	##Get lists of hyperparameters to try!
	extract_hyperparams , reconstruct_hyperparams , summary_sizes = get_params( domain , cross_pairs , debug )

	##Initialize a simulator and a pipeline or each random restart!  This will do the stuff for MT once
	simulators , pipelines = get_simulators_for_each_restart( domain , n_restarts , extract_hyperparams , 
								reconstruct_hyperparams , n_extraction_restarts , n_reconstruction_restarts , load_from_file )

	##Initialize all lists, and sets of single extraction parameters and their corresponding reconstruction parameters
	data = initialize_lists( extract_hyperparams , reconstruct_hyperparams , cross_pairs )

	##Compute the data to go into every run of the
	runs = compute_data_for_all_hyperparameter_runs( summary_sizes , data , pipelines , simulators , n_restarts )

	##For every summary, get the accuracy for each hyperparameter setting!
	run_outputs = compute_accuracy_and_value_for_each_parameter_setting_parallel( summary_sizes , data , simulators , pipelines , runs )

	##Combine back into dictionary
	data = build_data_dict_from_run_outputs( summary_sizes , data , run_outputs )

	if cross_pairs:
		##Save data used to generate the plot
		pickle.dump( data , open( "../plots/hyperparameter_selection_restarts/restart"+str(restart_number)+"_reconstruction_accuracy_grid_hyperparams_"+domain+"_data.p" , "wb" ) )
	else: ##Plot accuracies by summary size
		##Save data used to generate the plot
		pickle.dump( data , open( "../plots/hyperparameter_selection_restarts/restart"+str(restart_number)+"_reconstruction_by_summary_size_hyperparams_"+domain+"_data.p" , "wb" ) )

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument( '-r' , '--restart' , type=int )
	parser.add_argument( '-c' , '--crosspairs' , default=False , action='store_true' )
	parser.add_argument( '-d' , '--domain' , type=str , choices=[ "hivsimulator" , "gridworld" , "pacman" ] )
	parser.add_argument( '-l' , '--load' , default=False , action='store_true' )
	parser.add_argument( '--debug' , default=False , action='store_true' )
	args = parser.parse_args()

	start = time.time()
	run_results( args.domain , args.crosspairs , restart_number=args.restart , load_from_file=args.load , debug=args.debug )
	print( str( time.time() - start ) + " seconds\n\n" )


	
