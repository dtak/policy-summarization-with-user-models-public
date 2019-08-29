import os
import pickle
import scipy.stats
import numpy as np
import pylab as plt

##Set plotting parameters
plt.rcParams['lines.linewidth']=2.5
plt.rcParams['lines.markersize']=8
plt.rcParams['text.usetex']=True
plt.rcParams['text.latex.unicode']=True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['text.latex.preamble']= ['\usepackage{amsfonts}','\usepackage{amsmath}']
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize']=20
plt.rcParams['legend.fontsize']=20#12
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.rcParams['axes.titlesize']=20
plt.rcParams['figure.titlesize']=20
##Fonts non type 3
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def plot_all_hyperparameters( data ):

	cool_colors = [
		'aqua' , 
		'mediumaquamarine' ,
		'dodgerblue' , 
		'blue' ,
		'teal' ,
		'navy' ,
	]

	warm_colors = [
		'salmon' ,
		'red' ,
		'tomato' ,
		'maroon'
	]

	gammas = [ 0.1 , 1. ]
	degrees = [ 2 , 3 ]

	domains = sorted( data.keys() )
	performance_measures = [ "accuracies" , "values" ]

	#fig , ax = plt.subplots( 2 , 3 ) ##for 3 domains
	fig , ax = plt.subplots( 2 , 1 )
	for i in range( len( ax ) ):
		ax[ i ] = [ ax[ i ] ] ##So we can index into subplots

	##Set titles and axes
	for i , domain in enumerate( domains ):

		if domain == "gridworld":
			domain_name = "Random Gridworld"
		elif domain == "hivsimulator":
			domain_name = "HIV Simulator"
		else:
			assert domain == "pacman"
			domain_name = "PAC-MAN"
		ax[ 0 ][ i ].set_title( domain_name )

		#ax[ 1 ][ 1 ].set_xlabel( "Trajectory Length" ) ##For 3 domains
		ax[ 0 ][ 0 ].set_xlabel( "Trajectory Length" )
		for j , performance_measure in enumerate( performance_measures ):
			ax[ j ][ 0 ].set_ylabel( performance_measure.replace( "ies" , "y" ).replace( "s" , "" ).title() )

	##For legend
	handles = []
	names = []

	##Plot IL random lines, most common
	for i , domain in enumerate( domains ):
		for j , performance_measure in enumerate( performance_measures ):
			for k in range( len( data[ domain ][ "extract_names" ] ) ):
				extract_name = data[ domain ][ "extract_names" ][ k ]
				reconstruct_name = data[ domain ][ "reconstruct_names" ][ k ]
				if "random" in extract_name and "GRF" in reconstruct_name:

					##Reducing number of hyperparams
					valid_gamma = False
					for gamma in gammas:
						if "gamma," + str( gamma ) in reconstruct_name:
							valid_gamma = True

					valid_degree = False
					for degree in degrees:
						if "degree," + str( degree ) in reconstruct_name or "rbf" in reconstruct_name:
							valid_degree = True					

					if valid_gamma and valid_degree:
						handle,_,_ = ax[ j ][ i ].errorbar( data[ domain ][ "summary_sizes_list" ][ k ] , data[ domain ][ performance_measure+"_mean" ][ k ] 
														, yerr=data[ domain ][ performance_measure+"_95_ci" ][ k ] , color='grey' , alpha=1 , linestyle='--' )
	handles.append( handle )
	names.append( 'IL-Random' )

	##Plot IRL random lines
	for i , domain in enumerate( domains ):
		for j , performance_measure in enumerate( performance_measures ):
			for k in range( len( data[ domain ][ "extract_names" ] ) ):
				extract_name = data[ domain ][ "extract_names" ][ k ]
				reconstruct_name = data[ domain ][ "reconstruct_names" ][ k ]
				if "random" in extract_name and "IRL" in reconstruct_name:
					handle,_,_ = ax[ j ][ i ].errorbar( data[ domain ][ "summary_sizes_list" ][ k ] , data[ domain ][ performance_measure+"_mean" ][ k ] 
													, yerr=data[ domain ][ performance_measure+"_95_ci" ][ k ] , color='black' , alpha=1 , linestyle='--' )
	handles.append( handle )
	names.append( 'IRL-Random' )

	##Plot IL lines
	il_handles = []
	il_names = []
	for i , domain in enumerate( domains ):
		for j , performance_measure in enumerate( performance_measures ):
			for k in range( len( data[ domain ][ "extract_names" ] ) ):
				extract_name = data[ domain ][ "extract_names" ][ k ]
				reconstruct_name = data[ domain ][ "reconstruct_names" ][ k ]
				if "GRF" in extract_name:
					extract_name = extract_name.replace( "kernel," , "" ).replace( "gamma," , "" ).replace( "degree," , "" ).replace( "GRF" , "IL" )

					##Reducing number of hyperparams
					valid_gamma = False
					for gamma in gammas:
						if "gamma," + str( gamma ) in reconstruct_name:
							valid_gamma = True

					valid_degree = False
					for degree in degrees:
						if "degree," + str( degree ) in reconstruct_name or "rbf" in reconstruct_name:
							valid_degree = True					

					if valid_gamma and valid_degree:

						if i == 0 and j == 0:
							il_names.append( extract_name )
						color = cool_colors[ il_names.index( extract_name ) ]

						handle,_,_ = ax[ j ][ i ].errorbar( data[ domain ][ "summary_sizes_list" ][ k ] , data[ domain ][ performance_measure+"_mean" ][ k ] 
														, yerr=data[ domain ][ performance_measure+"_95_ci" ][ k ] , color=color , alpha=1 )
						if i == 0 and j == 0:
							il_handles.append( handle )

	##Plot IRL lines
	irl_handles = []
	irl_names = []
	for i , domain in enumerate( domains ):
		for j , performance_measure in enumerate( performance_measures ):
			for k in range( len( data[ domain ][ "extract_names" ] ) ):
				extract_name = data[ domain ][ "extract_names" ][ k ]
				reconstruct_name = data[ domain ][ "reconstruct_names" ][ k ]
				if "IRL" in extract_name:
					extract_name = extract_name.replace( "trajlen" , "" )
					if i == 0 and j == 0:
						irl_names.append( extract_name )
					color = warm_colors[ irl_names.index( extract_name ) ]
					handle,_,_ = ax[ j ][ i ].errorbar( data[ domain ][ "summary_sizes_list" ][ k ] , data[ domain ][ performance_measure+"_mean" ][ k ] 
													, yerr=data[ domain ][ performance_measure+"_95_ci" ][ k ] , color=color , alpha=1 )
					if i == 0 and j == 0:
						irl_handles.append( handle )


	handles = il_handles + irl_handles + handles
	names = il_names + irl_names + names

	plt.legend( handles , names , loc='center left' , bbox_to_anchor=(1, 1), ncol=1 , columnspacing=0. , handletextpad=0. )
	plt.tight_layout( pad=0, w_pad=-1., h_pad=0 )
	plt.savefig( "../plots/hyperparameter_selection_restarts/reconstruction_by_summary_size_hyperparams.pdf" , bbox_inches="tight" )
	#plt.show()

def plot_summary_comparison_grids( data ):

	plt.rcParams['font.size']=12
	plt.rcParams['axes.labelsize']=12
	plt.rcParams['legend.fontsize']=12
	plt.rcParams['xtick.labelsize']=12
	plt.rcParams['ytick.labelsize']=12
	plt.rcParams['axes.titlesize']=12
	plt.rcParams['figure.titlesize']=12

	domains = sorted( data.keys() )
	performance_measures = [ "accuracies" , "values" ]

	##fig , ax = plt.subplots( 2 , 3 ) ##For all 3 domains
	fig , ax = plt.subplots( 2 , 1 )
	for i in range( len( ax ) ):
		ax[ i ] = [ ax[ i ] ] ##So we can index into subplots

	##Set titles and axes
	for i , domain in enumerate( domains ):

		if domain == "gridworld":
			domain_name = "Random Gridworld"
		elif domain == "hivsimulator":
			domain_name = "HIV Simulator\n."
		else:
			assert domain == "pacman"
			domain_name = "PAC-MAN\n."

		ax[ 0 ][ i ].set_title( domain_name )

		extract_names = sorted( np.unique( data[ domain ][ "extract_names" ] ) )
		##Format for plot
		extract_labels = []
		for j in range( len( extract_names ) ):
			extract_labels.append( extract_names[ j ].replace( "kernel," , "" ).replace( "gamma," , "\n" ).replace( "degree," , "" ).replace( "GRF" , "IL" ).replace( "trajlen" , "" ).replace( "rollhor" , "" ) ) #.replace( ",0." , "\n0." ).replace( "rbf," , "rbf\n" ) )

		reconstruct_names = sorted( np.unique( data[ domain ][ "reconstruct_names" ] ) )
		##Format for plot
		reconstruct_labels = []
		for j in range( len( reconstruct_names ) ):
			reconstruct_labels.append( reconstruct_names[ j ].replace( "kernel," , "" ).replace( "gamma," , "\n" ).replace( "degree," , "" ).replace( "GRF" , "IL" ).replace( "trajlen" , "" ).replace( "rollhor" , "" ).replace( "-10" , "" ).replace( "-25" , "" )  )

		for j , performance_measure in enumerate( performance_measures ):

			if performance_measure == "accuracies":
				ax[ j ][ 0 ].set_ylabel( "Accuracy" , fontsize=10 )
			else:
				ax[ j ][ 0 ].set_ylabel( "Value Diff." , fontsize=10 )

			performance_grid = np.zeros( ( len( reconstruct_names ) , len( extract_names ) ) )
			for r , reconstruct_name in enumerate( reconstruct_names ):
				for e , extract_name in enumerate( extract_names ):
					for k in range( len( data[ domain ][ performance_measure+"_mean" ] ) ):
						if data[ domain ][ "extract_names" ][ k ] == extract_name and data[ domain ][ "reconstruct_names" ][ k ] == reconstruct_name:
							performance_grid[ r , e ] = data[ domain ][ performance_measure+"_mean" ][ k ][ 0 ]

			if j == 1:
				ax[ j ][ i ].set_xticks( np.arange( len( extract_labels ) ) )
				ax[ j ][ i ].set_xticklabels( extract_labels , rotation=90 , fontsize=10 )
			else:
				ax[ j ][ i ].set_xticks( np.arange( len( extract_labels ) ) )
				ax[ j ][ i ].set_xticklabels( [ '' ] * len( extract_labels ) )

			ax[ j ][ i ].set_yticks( np.arange( len( reconstruct_labels ) ) )
			ax[ j ][ i ].set_yticklabels( reconstruct_labels , fontsize=10 )
			
			if performance_measure == "accuracies":
				colors = ax[ j ][ i ].imshow( performance_grid , cmap='OrRd' )
			else:
				##Min max scaling
				if performance_measure == "values":
					min_val = np.min( performance_grid )
					max_val = np.max( performance_grid )
					performance_grid = ( performance_grid - min_val ) / ( max_val - min_val )
				colors = ax[ j ][ i ].imshow( performance_grid , cmap='GnBu' )
			lower_lim = np.ceil( np.min( performance_grid ) * 10. ) / 10.
			upper_lim = np.floor( np.max( performance_grid ) * 10. ) / 10.
			cb = plt.colorbar( colors , ax=ax[ j ][ i ] , aspect=5 , fraction=.1 , ticks=[ lower_lim , upper_lim ] )
			cb.ax.tick_params( labelsize=10 )

			#fig.text(0.5, 0.20, 'Extraction', ha='center') ##For 3 domains
			#fig.text(-0.05, 0.5, 'Reconstruction', va='center', rotation='vertical') ##For 3 domains
			#ax[ 0 ][ 0 ].set_xlabel( 'Extraction' )
			fig.text(0.6, 0.47, 'Extraction', ha='center') ##For 3 domains
			fig.text(0.25, 0.5, 'Reconstruction', va='center', rotation='vertical')

	#plt.tight_layout( pad=0, w_pad=1., h_pad=-12. ) ##For 3 domains

	plt.savefig( "../plots/hyperparameter_selection_restarts/performance_table.pdf" , bbox_inches="tight" )


if __name__ == "__main__":

	n_runs = 1 ##Number of random restarts you've run of the hyperparameter comparison pipeline

	data_dir = "../plots/hyperparameter_selection_restarts/"

	file_identifiers = [ "reconstruction_by_summary_size_hyperparams_" , "reconstruction_accuracy_grid_hyperparams_" ]

	##Collect data from diff domains and restarts into 1 dictionary
	for f in range( len( file_identifiers ) ):

		data = {}
		for fname in os.listdir( data_dir ):

			if file_identifiers[ f ] in fname and "_data.p" in fname:

				restart_data = pickle.load( open( data_dir + fname , "rb" ) )
				domain = fname.split( file_identifiers[ f ] )[ 1 ].split( "_data.p" )[ 0 ]
				if domain not in data:
						data[ domain ] = restart_data.copy()
						for i in range( len( data[ domain ][ "extract_names" ] ) ):
							for j in range( len( data[ domain ][ "summary_sizes_list" ][ i ] ) ):
								data[ domain ][ "accuracies_mean" ][ i ][ j ] = [ data[ domain ][ "accuracies_mean" ][ i ][ j ] ]
								data[ domain ][ "values_mean" ][ i ][ j ] = [ data[ domain ][ "values_mean" ][ i ][ j ] ]
				else:
					for i in range( len( data[ domain ][ "extract_names" ] ) ):
						found = False
						for j in range( len( restart_data[ "extract_names" ] ) ):
							if data[ domain ][ "extract_names" ][ i ] == restart_data[ "extract_names" ][ j ] and \
								data[ domain ][ "reconstruct_names" ][ i ] == restart_data[ "reconstruct_names" ][ j ] and \
								data[ domain ][ "summary_sizes_list" ][ i ] == restart_data[ "summary_sizes_list" ][ j ]:
								for k in range( len( data[ domain ][ "summary_sizes_list" ][ i ] ) ):
									data[ domain ][ "accuracies_mean" ][ i ][ k ].append( restart_data[ "accuracies_mean" ][ j ][ k ] )
									data[ domain ][ "values_mean" ][ i ][ k ].append( restart_data[ "values_mean" ][ j ][ k ] )
								assert not found
								found = True
						assert found

		##Compute summary statistics for performance metrics in each domain
		for domain in data.keys():
			for performance_measure in [ "accuracies" , "values" ]:
				data[ domain ][ performance_measure+"_mean" ] = np.array( data[ domain ][ performance_measure+"_mean" ] )[ : , : , :n_runs ]

				data[ domain ][ performance_measure+"_std_dev" ] = np.std( data[ domain ][ performance_measure+"_mean" ] , axis=2 )
				data[ domain ][ performance_measure+"_95_ci" ] = 1.96 * scipy.stats.sem( data[ domain ][ performance_measure+"_mean" ] , axis=2 )
				data[ domain ][ performance_measure+"_mean" ] = np.mean( data[ domain ][ performance_measure+"_mean" ] , axis=2 )

		if "accuracy" in file_identifiers[ f ]: ##Plot performance grids
			plot_summary_comparison_grids( data )
		else: ##Plot hyperparameter search
			plot_all_hyperparameters( data )

