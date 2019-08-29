import numpy as np
import pylab as plt
from simulator import Simulator

import os
import sys
sys.path.append( os.getcwd() + '/models/' )
from value_iteration import find_policy

class RandomGridworldSimulator( Simulator ):

	def __init__( self , max_x=5 , max_y=5 , n_features=5 , discount=0.99 , random=True , n_value_simulations=1 , n_value_steps=20 ):

		self.domain = "random_gridworld"

		self.action_names = [ 'Stay' , 'Up' , 'Down' , 'Left' , 'Right' ]

		self.rollout_horizon = 10
		self.discount = discount

		self.n_actions = 5

		##Save these for later use in comparing policy values!!!
		self.n_value_simulations = n_value_simulations
		self.n_value_steps = n_value_steps

		assert self.rollout_horizon == self.n_value_steps

		##Grid size
		self.max_x = max_x
		self.max_y = max_y
		self.n_states = max_x * max_y ##Size of the grid

		##x y state_identifiers
		self.state_identifiers = []
		for i in range( self.max_x ):
			for j in range( self.max_y ):
				self.state_identifiers.append( [ i , j ] )

		##Features are binary indicators for (gridsize - 1) different colors - each cell gets a random color -- could be changed to any number
		self.n_features = n_features #np.random.choice( np.arange( 3 , 7 ) )
		self.feature_names = ["Feat Color " + str(i) for i in range(self.n_features)]

		##Randomly set reward weights on colors!
		self.reward_weights = self.get_reward_weights(random)

		self.colors = []
		for x , y in self.state_identifiers:
			self.colors.append( self.get_color_vector( x , y ) )
		self.colors = np.array( self.colors )

		self.neighbors = []
		for i in range( self.colors.shape[ 0 ] ):
			self.neighbors.append( self.get_next_state_feature_vector( i ) )
		self.neighbors = np.array( self.neighbors )
		#assert False , "Change back features!!"
		self.features = np.concatenate( ( self.colors , self.neighbors ) , axis=1 )

		##Transition probabilities for all states - deterministic for now
		self.transition_probas = self.get_transition_probabilities()

		self.valid_actions = {}
		for i in range( self.max_x ):
			for j in range( self.max_y ):
				allowable_actions = [ 'Stay' ]
				if i != 8:
					allowable_actions.append( 'Right' )
				if i != 0: ##Don't turn right if Right wall
					allowable_actions.append( 'Left' )
				if j != 0: ##Don't go down if Bottom wall
					allowable_actions.append( 'Down' )
				if j != 8: ##Don't go down if Bottom wall
					 allowable_actions.append( 'Up' )
				allowable_action_indices = []
				for action in allowable_actions:
					allowable_action_indices.append( self.action_names.index( action ) )
				self.valid_actions[ self.state_identifiers.index( [ i , j ] ) ] = allowable_action_indices

		##Get optimal policy for selected reward weights and features + indices of states with multiple optimal actions
		self.policy, self.multiple_optimal_indices = self.get_policy()

		##Set original policy value
		self.policy_value = self.get_policy_value(policy=self.policy, n_simulations=n_value_simulations, n_steps=n_value_steps)

		##Feature indices for IL and IRL
		self.irl_indices = np.arange( self.colors.shape[ 1 ] ).tolist()
		self.il_indices = np.arange( self.colors.shape[ 1 ] , self.features.shape[ 1 ] ).tolist()

		super( RandomGridworldSimulator , self ).__init__( )

		self.start_states = np.arange( len( self.unique_state_identifiers ) )

	def get_color_vector( self , x , y ):

		##Feature vector of length n_features - representing n_features colors
		feature_vector = np.zeros(self.n_features)

		##Randomly select a color for cell
		color = np.random.choice(self.n_features)

		feature_vector[color] = 1.0

		return feature_vector

	def get_next_state_feature_vector( self , current_state ):
		##Feature vector of length n_features - representing n_features colors
		feature_vector = []
		##For all possible actions
		for action in range( self.n_actions ):
			#Get the next state
			next_state = self.get_next_state( current_state , action )
			#Get the next state feature vector
			feature_vector += list( self.colors[ next_state ] )

		return np.array( feature_vector )

	##given a state and an action get index of next state from state list
	def get_next_state( self , current_state , action ):

		current_state_identifiers = self.state_identifiers[ current_state ] ##state_identifiers = [row, col]
		next_state_identifiers = [ current_state_identifiers[ 0 ] , current_state_identifiers[ 1 ] ] ##Initialize to current state_identifiers

		##Stay
		if action == 0:
			pass
		##Up
		elif action == 1:
			new_y = current_state_identifiers[ 1 ] + 1
			if new_y < self.max_y:
				next_state_identifiers[ 1 ] = new_y
		##Down
		elif action == 2:
			new_y = current_state_identifiers[ 1 ] - 1
			if new_y >= 0:
				next_state_identifiers[ 1 ] = new_y
		##Left
		elif action == 3:
			new_x = current_state_identifiers[ 0 ] - 1
			if new_x >= 0:
				next_state_identifiers[ 0 ] = new_x
		##Right
		elif action == 4:
			new_x = current_state_identifiers[ 0 ] + 1
			if new_x < self.max_x:
				next_state_identifiers[ 0 ] = new_x
		else:
			assert False , "Grid world only has 5 actions!"

		next_state = self.state_identifiers.index( next_state_identifiers )

		return next_state

	##Get probability of moving ftom state i to state k under action a (deterministic)
	def get_transition(self, state_i, state_k, action):

		next_state = self.get_next_state(state_i, action)

		##state_k is the intended state to move to
		if next_state == state_k:
			return 1.0
		##illegal action in state (takes out of grid) - stay in origin state
		elif next_state == state_i and state_i == state_k:
			return 1.0
		else:
			return 0.0

	##Get transition probabilities for all states
	def get_transition_probabilities(self):

		transition_probability = np.array(
			[[[self.get_transition(state_i, state_k, action_a)
			   for state_k in range(self.n_states)]
			  for action_a in range(self.n_actions)]
			 for state_i in range(self.n_states)])

		return transition_probability

	def get_reward_weights(self, random):

		if random:
			##Get random reward weights
			reward_weights = np.zeros(self.n_features)

			##Sample reward weights random uniformly from L1 unit ball of dimension n_features
			##Note: This is how it was implemented in the code of the Brown & Niekum paper so I followed it
			for i in range(self.n_features):
				rand = np.random.uniform()
				if rand < 0.5:
					reward_weights[i] = np.log(2.0 * rand)
				else:
					reward_weights[i] = -np.log(2.0 - 2.0 * rand)

			##Normalize reward weights
			abs_sum = np.sum(np.absolute(reward_weights))
			reward_weights /= abs_sum
		else:
			##Sparse reward weights for user study
			reward_weights = np.array([100.0, 10.0, 0.0, -10.0, -100.0])
		return reward_weights

	def get_policy(self):

		##Calculate optimal policy for gridworld using value iteration
		rewards = np.dot(self.colors, self.reward_weights)

		policy, multiple_state_indices = find_policy(self.n_states, self.n_actions, self.transition_probas, rewards,
													 discount=0.9, stochastic=False, valid_actions=self.valid_actions, return_multiple=True)
		return policy, multiple_state_indices

	def extract_trajectories( self , n_steps=1 ):

		##Extract a trajectory of length n_steps starting from each unique state in the grid
		##We will call this function with n_steps = 1 to avoide duplicate states
		trajectories = []
		for i in range( self.n_states ):
			current_state = i

			trajectory = []
			for j in range( n_steps ): ##Play out policy for n_steps steps
				action = self.policy[ current_state ] ##Get the action corresponding to this state

				next_state = self.get_next_state( current_state , action )

				onehot_action = np.zeros( self.n_actions )
				onehot_action[ action ] = 1
				onehot_action = onehot_action.tolist()

				trajectory.append( [ self.features[ current_state ] , onehot_action , self.state_identifiers[ current_state ] , np.dot( self.reward_weights , self.features[ next_state ] ) ] )

				current_state = next_state

			trajectories.append( trajectory ) ##Save the trajectories of states and actions

		return trajectories

	def run_mc_episode( self , state , action ):

		episode = [ self.features[ state ] ]

		for i in range( self.rollout_horizon - 1 ):
			
			state = self.get_next_state( state , action )
			action = self.policy[ state ]
			
			episode.append( self.features[ state ] )

		return np.array( episode )


	def get_state_action_fcounts( self ):

		discount_matrix = np.tile( np.array( [ self.discount ** i for i in range( self.rollout_horizon ) ] ) , ( self.n_features , 1 ) ).T

		##Get feature counts for all unique states and actions in grid
		feature_counts = {}
		for state in range( self.n_states ):
			##For each action in state run mc rollout in simulator and calculate expected feature counts mu_sa (defined under eqation 4 in paper)
			for action in range( self.n_actions ):

				sa_features = self.run_mc_episode( state , action )[ : , self.irl_indices ]
				sa_fcounts = np.multiply( sa_features , discount_matrix[ :sa_features.shape[ 0 ] , : ] )
				mu_sa = np.sum( sa_fcounts , axis=0 )

				##Add feature counts to dicationary
				feature_counts[ ( state , action ) ] = mu_sa

		return feature_counts

	def get_candidate_trajectories( self , trajectory_length ):

		candidate_trajectories = []  ##(List of state features, List of actions, List of identifiers , List of rewards) for each trajectory

		##Generate a trajectory of length trajectory_length from each state in the grid following policy
		for i in range(self.n_states):
			current_state = i

			trajectory_features = []  ##List of features in a trajectory
			trajectory_actions = []  ##List of actions in a trajectory
			trajectory_identifiers = []  ##List of state indices in trajectory
			for j in range(trajectory_length):  ##Play out policy for trajectory_length steps
				action = self.policy[current_state]  ##Get the action corresponding to this state

				trajectory_features.append(self.features[current_state])
				trajectory_actions.append(action)
				trajectory_identifiers.append(self.state_identifiers[current_state])

				current_state = self.get_next_state(current_state, action)

			candidate_trajectories.append((trajectory_features, trajectory_actions, trajectory_identifiers))

		return candidate_trajectories

	def get_policy_value(self, policy, n_simulations=1, n_steps=20):

		n_simulations = 100

		##Get value of policy by average of rewards sum from running n_simulations of length n_steps from each state
		policy_value = 0
		for state in range(len(policy)):
			state_value = 0

			for sim in range(n_simulations):
				current_state = state
				sum_rewards = 0

				for i in range(n_steps):
					action = policy[current_state]
					current_state = self.get_next_state(current_state, action)
					sum_rewards += np.dot(self.reward_weights, self.colors[current_state])*(self.discount**i)

				state_value += sum_rewards

			policy_value += (float(state_value)/float(n_simulations))

		return policy_value/float(len(policy))

	def plot_policy( self ):
		self.plot_gridworld( self.policy , self.state_identifiers , self.features , self.feature_names , plot_name="random_gridworld_policy" )

	def plot_summary( self , selected_points , plot_name , plot_dir="../plots/randomgridworld_summaries/" ):
		self.plot_gridworld( self.policy , self.state_identifiers , self.features , self.feature_names , selected_points=selected_points , plot_name=plot_name , plot_dir=plot_dir )

	def plot_summaries( self , summaries , extract_names , plot_dir="../plots/randomgridworld_summaries/" ):
		##IKE: Updaeted to work with 
		for i in range( len( summaries ) ):
			selected_points = []
			summary_length = len(summaries[i][0])
			for j , summary in enumerate( summaries[ i ][0] ):
				for idx in summary:

					##Get state state_identifiers from state_identifiers list
					if len( summary ) > 1:
						selected_points.append( self.state_identifiers[ idx ] + [ j ] )
					else:
						selected_points.append( self.state_identifiers[ idx ] + [ 0 ] )

			if "GRF" in extract_names[ 0 ][ i ]:
				plot_name = "gridworld_il_summary"
			elif "POLICY" in extract_names[ 0 ][ i ]:
				plot_name = "gridworld_full_policy"
			else:
				assert "IRL" in extract_names[ 0 ][ i ] , str( extract_names[ 0 ][ i ] )
				plot_name = "gridworld_irl_summary"

			self.plot_summary( selected_points , plot_name=plot_name , plot_dir=plot_dir )

	def plot_gridworld( self , policy , states , features , feature_names , plot_name="" , selected_features=[] , selected_points=[] , plot_dir="../plots/randomgridworld_summaries/" ):

		print( plot_name )

		if len( selected_points ):
			num_trajectories = np.max( np.array( selected_points )[ : , 2 ] )
		else:
			num_trajectories = 1

		##Plotting summary or original policy?
		if selected_points:
			#plt.title( plot_name.title() , y=1.06)
			points_to_plot = selected_points
			plot_rewards = False
		else:
			plt.title( "Original Policy with True Reward Values" , y=1.06)
			points_to_plot = np.concatenate( ( states , np.zeros( ( len( states ) , 1 ) ) ) , axis=1 ).tolist()
			plot_rewards = True

		##Set grid tile colors
		color_grid = np.zeros( ( self.max_x , self.max_y ) )
		for i, state in enumerate(states):
			##For original policy plot cells by reward value
			if plot_rewards:
				color_grid[state[0], state[1]] = np.dot(features[i], self.reward_weights)
			##For summaries plot by feature indices
			else:
				color_grid[state[0], state[1]] = np.argmax(features[i])

		#plt.imshow(np.swapaxes(color_grid, 0, 1))
		plt.imshow(np.swapaxes(color_grid, 0, 1), origin="lower", vmin=0 , vmax=self.n_features, cmap='tab10')

		##For original policy - plot a color bar showing values of true rewards per state
		if plot_rewards:
			cbar = plt.colorbar()
			cbar.set_label('Reward Value')

		colors = [
			'red' ,
			'Yellow' ,
			'Chartreuse' ,
			'Aqua' ,
			'Navy' ,
			'White' ,
		]


		##Plot policy actions
		#action_markers = ["H", "^", "v", "<", ">"]
		action_deltas = [ ( 0 , 0 ) , ( 0 , 0.8 ) , ( 0 , -0.8 ) , ( -0.8 , 0 ) , ( 0.8 , 0 ) ]
		collected_trajectories = []
		labels = []
		lines = []
		for state in points_to_plot:
			state_idx = states.index(state[ :2 ])
			action = policy[state_idx]
			
			x_delta = action_deltas[action][0]
			y_delta = action_deltas[action][1]
			if x_delta == 0:
				x_start = state[0] + ( ( np.random.rand() - 0.5 ) * 0.25 )
			else:
				x_start = state[0] - ( x_delta / 2. )
			if y_delta == 0:
				y_start = state[1] + ( ( 0.5 - np.random.rand() ) * 0.25 )
			else:
				y_start = state[1] - ( y_delta / 2. )

			if action > 0:
				marker = plt.arrow(x_start, y_start, x_delta, y_delta, width=0.1, length_includes_head=True, edgecolor='k' , facecolor=colors[int(state[2])-1] , alpha=0.8 )
			else:
				marker = plt.plot(x_start, y_start, color=colors[int(state[2])-1], marker="H", markersize=12 , markeredgecolor='k' , alpha=0.8)

		##Plot grid lines
		ax = plt.gca()
		# Minor ticks
		ax.set_xticks(np.arange(-.5, self.max_x, 1), minor=True)
		ax.set_xticklabels(['']*len(np.arange(-.5, self.max_x, 1)))
		ax.set_yticks(np.arange(-.5, self.max_y, 1), minor=True)
		ax.set_yticklabels(['']*len(np.arange(-.5, self.max_x, 1)))

		# Gridlines based on minor ticks
		ax.grid(which='minor', color='k')#, linestyle='-', linewidth=2)

		if plot_name:
			plt.savefig( plot_dir+plot_name+".png" , bbox_inches="tight" )
		else:
			plt.show()
		plt.clf()
		plt.close()

	def plot_prediction_states_on_summaries( self , prediction_states , summaries , extract_names , prefix , plot_dir="../plots/randomgridworld_summaries/prediction_states/" ):
		
		##Clear out directory
		for del_fname in os.listdir( plot_dir ):
			if prefix in del_fname:
				os.remove( plot_dir + del_fname )

		##IKE: Updaeted to work with 
		for i in range( len( summaries ) ):
			selected_points = []
			summary_length = len(summaries[i][0])
			for j , summary in enumerate( summaries[ i ][0] ):
				for idx in summary:

					##Get state state_identifiers from state_identifiers list
					if len( summary ) > 1:
						selected_points.append( self.state_identifiers[ idx ] + [ j ] )
					else:
						selected_points.append( self.state_identifiers[ idx ] + [ 0 ] )

			if "GRF" in extract_names[ 0 ][ i ]:
				if prefix == "test":
					plot_name = "test_il"
				elif prefix == "practice":
					plot_name = "practice_il"
				else:
					assert False
			elif "IRL" in extract_names[ 0 ][ i ]:
				if prefix == "test":
					plot_name = "test_irl"
				elif prefix == "practice":
					plot_name = "practice_irl"
				else:
					assert False
			else:
				assert False , str( extract_names[ 0 ][ i ] )

			self.plot_prediction_states_on_summary( prediction_states , selected_points , plot_name=plot_name , plot_dir=plot_dir )

	def plot_prediction_states_on_summary( self , prediction_states , selected_points ,  plot_name , plot_dir="../plots/randomgridworld_summaries/prediction_states/" ):

		if len( selected_points ):
			num_trajectories = np.max( np.array( selected_points )[ : , 2 ] )
		else:
			num_trajectories = 1

		colors = [
			'red' ,
			'Yellow' ,
			'Chartreuse' ,
			'Aqua' ,
			'Navy' ,
			'White' ,
		]

		action_deltas = [ ( 0 , 0 ) , ( 0 , 0.8 ) , ( 0 , -0.8 ) , ( -0.8 , 0 ) , ( 0.8 , 0 ) ]

		##Set grid tile colors
		for i, state in enumerate( self.state_identifiers ):
			##Colored square
			color_grid = [ [ np.argmax(self.colors[i]) ] ]

			plt.clf()
			plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
			plt.imshow(np.swapaxes(color_grid, 0, 1), origin="lower", vmin=0 , vmax=self.n_features , cmap='tab10')
			
			##Do arrows
			action = self.policy[i]
			in_selected_points = False
			for point in selected_points:
				if point[ :2 ] == state:

					x_delta = action_deltas[action][0]
					y_delta = action_deltas[action][1]
					if x_delta == 0:
						x_start = ( ( np.random.rand() - 0.5 ) * 0.25 )
					else:
						x_start = -( x_delta / 2. )
					if y_delta == 0:
						y_start = ( ( 0.5 - np.random.rand() ) * 0.25 )
					else:
						y_start = -( y_delta / 2. )

					if action > 0:
						marker = plt.arrow(x_start, y_start, x_delta, y_delta, width=0.05, length_includes_head=True, edgecolor='k' , facecolor=colors[int(point[2])-1] , alpha=0.8 )
					else:
						marker = plt.plot(x_start, y_start, color=colors[int(point[2])-1], marker="H", markersize=64, markeredgecolor='k', alpha=0.8)

					in_selected_points = True

			state_plot_name = plot_name + "_" + str( i )

			if i in prediction_states:
				assert not in_selected_points , "Can't be both prediction and summary state!"
				plt.plot( [ -.5 , -.5 ] , [ -.5 , .45 ] , linestyle='-' , color='k' , linewidth=50 )
				plt.plot( [ -.5 , .45 ] , [ .45 , .45 ] , linestyle='-' , color='k' , linewidth=50 )
				plt.plot( [ .45 , .45 ] , [ .45 , -.5 ] , linestyle='-' , color='k' , linewidth=50 )
				plt.plot( [ .45 , -.5 ] , [ -.5 , -.5 ] , linestyle='-' , color='k' , linewidth=50 )

				state_plot_name += "_predict"

			plt.savefig( plot_dir+state_plot_name+".png" , bbox_inches="tight" , pad_inches=0 )
			plt.close()

		
