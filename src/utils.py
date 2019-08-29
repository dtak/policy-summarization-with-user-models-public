import numpy as np

def find_corresponding_reconstruction( extract_hyperparams , reconstruct_hyperparams_list ):
	##Find the corresponding reconstruction hyperparams
	self_reconstruction_hyperparams = None
	for reconstruct_hyperparams in reconstruct_hyperparams_list:
		if ( ( "feature_indices" not in extract_hyperparams and 
			"feature_indices" not in reconstruct_hyperparams ) or
			np.array_equal( extract_hyperparams[ "feature_indices" ] , reconstruct_hyperparams[ "feature_indices" ] ) ):
			if extract_hyperparams[ "model" ] == "irl" and reconstruct_hyperparams[ "model" ] == "irl" and \
				 ( not reconstruct_hyperparams[ "match_rollout" ] or \
				 	extract_hyperparams[ "trajectory_length" ] == reconstruct_hyperparams[ "maxent_params" ][ "rollout_horizon" ] ):
				assert self_reconstruction_hyperparams == None , str( extract_hyperparams ) + " - " + str( self_reconstruction_hyperparams )##Only assigned once
				self_reconstruction_hyperparams = reconstruct_hyperparams
			if extract_hyperparams[ "model" ] != "irl" and reconstruct_hyperparams[ "model" ] != "irl" and extract_hyperparams[ "kernel_params" ] == reconstruct_hyperparams[ "kernel_params" ]:
				assert self_reconstruction_hyperparams == None , str( extract_hyperparams ) + " - " + str( self_reconstruction_hyperparams ) ##Only assigned once
				self_reconstruction_hyperparams = reconstruct_hyperparams
	assert self_reconstruction_hyperparams != None , "No reconstruction hyperparams found!! " + str( extract_hyperparams )
	return self_reconstruction_hyperparams

def get_model_name( hyperparams ):
	if "algorithm" in hyperparams and hyperparams[ "algorithm" ] == "random": #Rand summary
		return "random-length"+str( hyperparams[ "trajectory_length" ] ) 

	model_name = hyperparams[ "model" ].upper() 
	if "feature_indices" in hyperparams:
		model_name = model_name + "-featidx" + ",".join( [ str( i ) for i in hyperparams[ "feature_indices" ] ] )
	if "kernel_params" in hyperparams:
		model_name = model_name + "-" + ",".join( [ key + "," + str( val ) for key , val in hyperparams[ "kernel_params" ].items() ] )
	if "trajectory_length" in hyperparams:
		model_name = model_name + "-trajlen" + str( hyperparams[ "trajectory_length" ] )
	if "maxent_params" in hyperparams:
		model_name = model_name + "-rollhor" + str( hyperparams[ "maxent_params" ][ "rollout_horizon" ] )
	
	return model_name
