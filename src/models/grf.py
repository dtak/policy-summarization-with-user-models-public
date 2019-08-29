import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import polynomial_kernel

from sklearn.preprocessing import StandardScaler

class GRF:

	def __init__( self , kernel_params , n_classes , fit_kernel=False , X=None , y=None ):

		if kernel_params[ "kernel" ] == "rbf":
			kernel = rbf_kernel
		elif kernel_params[ "kernel" ] == "poly":
			kernel = polynomial_kernel
		elif kernel_params[ "kernel" ] == "linear":
			kernel = linear_kernel
		elif kernel_name == "matern":
			assert False
			kernel = Matern()
		elif kernel_name == "rq":
			assert False
			kernel == RationalQuadratic()
		else:
			assert False , "Invalid kernel in grf: " + str( kernel_name )

		self.kernel_params = {}
		for key , val in kernel_params.items():
			if key != "kernel":
				self.kernel_params[ key ] = val

		self.n_classes = n_classes
		
		self.kernels = []
		if fit_kernel:
			assert False
			assert X is not None and y is not None , "If you want to fit kernels, you must pass in X and y!"
		
			binary_class_labels = []
			for k in range( self.n_classes ):
				binary_labels = np.zeros( y.shape[ 0 ] )
				binary_labels[ np.where( y == k) ] = 1
				if len( np.unique( binary_labels ) ) > 1:
					##Only fit if this will actually be used in the end
					gp = GaussianProcessClassifier( kernel=kernel )
					gp.fit( X , binary_labels )
					self.kernels.append( gp.kernel_ )
				else:
					self.kernels.append( kernel )
		else:
			for k in range( self.n_classes ):
				#kernel.set_params( **self.kernel_params )
				self.kernels.append( kernel )

	def predict_proba_binary( self , laplacian , labeled , unlabeled , labels ):

		self.l_uu = laplacian[ unlabeled , : ][ : , unlabeled ]
		
		try:
			self.l_uu_inv = np.linalg.inv( self.l_uu )
		except:
			print( len( labeled ) )
			print( np.unique( np.ravel( self.l_uu ) ) )
			assert False

		self.l_ul = laplacian[ unlabeled , : ][ : , labeled ]
		self.unlabeled_prev = unlabeled
		f_u = np.matmul( np.matmul( ( -1 * self.l_uu_inv ) , self.l_ul ) , labels )
		return f_u

	def predict_probas( self , X , y , unlabeled ):

		self.X = X
		self.y = y

		self.binary_class_labels = []
		for k in range( self.n_classes ):
			binary_labels = np.zeros( ( y.shape[ 0 ] , 2 ) )
			binary_labels[ np.where( y == k) , 1 ] = 1
			binary_labels[ np.where( y != k) , 0 ] = 1
			self.binary_class_labels.append( binary_labels )
		self.binary_class_labels = np.array( self.binary_class_labels )

		labeled = np.setdiff1d( np.arange( self.X.shape[ 0 ] ) , unlabeled )
		
		all_class_probas = []
		for i in range( self.n_classes ):
			binary_labels = self.binary_class_labels[ i ][ labeled , : ]
			laplacian = self.compute_laplacian( self.kernels[ i ] , X )
			class_probas = self.predict_proba_binary( laplacian , labeled , unlabeled , binary_labels )[ : , 1 ]

			all_class_probas.append( class_probas )
		all_class_probas = np.swapaxes( np.array( all_class_probas ) , 0 , 1 )
		self.f_u = all_class_probas
		self.unlabeled = unlabeled

		return all_class_probas

	def predict( self , X , y , unlabeled ,  test=[] , fit_kernel=False ):

		all_class_probas = self.predict_probas( X , y , unlabeled )

		##Ike: Stochastic!
		predictions = []
		for class_probas in all_class_probas:
			predictions.append( np.random.choice( np.where( class_probas == np.max( class_probas ) )[ 0 ] ) )

		return np.array( predictions )

	def predict_proba_binary_update( self , query_idx , f_u , labels ):

		unlabeled_query_idx = list( self.unlabeled ).index( query_idx ) ##Get index of query id in unlabeled array!
		f_u = f_u + ( np.argmax( labels[ query_idx ] ) - f_u[ unlabeled_query_idx ] ) * ( self.l_uu_inv[ : , unlabeled_query_idx ] / self.l_uu_inv[ unlabeled_query_idx , unlabeled_query_idx ] )
		return f_u , unlabeled_query_idx

	def predict_with_new_label( self , query_idx ):
		assert self.f_u is not None , 'Must predict before calling predict_with_new_label'

		all_class_probas = []
		for i in range( len( self.binary_class_labels ) ):
			f_u , unlabeled_query_idx = self.predict_proba_binary_update( query_idx , self.f_u[ : , i ] , self.binary_class_labels[ i ] )
			f_u = np.delete( f_u , unlabeled_query_idx )
			all_class_probas.append( f_u )
		all_class_probas = np.swapaxes( np.array( all_class_probas ) , 0 , 1 )

		predictions = np.argmax( all_class_probas , axis=1 )

		return predictions

	def predict_with_new_trajectory( self , query_indices ):
		assert self.f_u is not None , 'Must predict before calling predict_with_new_label'

		all_class_probas = []
		for i in range( len( self.binary_class_labels ) ):
			f_u = np.copy( self.f_u[ : , i ] )
			unlabeled_query_indices = []
			for query_idx in query_indices:
				f_u , unlabeled_query_idx = self.predict_proba_binary_update( query_idx , f_u , self.binary_class_labels[ i ] )
				unlabeled_query_indices.append( unlabeled_query_idx )
			f_u = np.delete( f_u , unlabeled_query_indices )
			all_class_probas.append( f_u )
		all_class_probas = np.swapaxes( np.array( all_class_probas ) , 0 , 1 )

		predictions = np.argmax( all_class_probas , axis=1 )

		return predictions

###################################################################HELPERS##################################################################################

	def compute_laplacian( self , kernel , X ):
		G = kernel( X , **self.kernel_params )

		D = np.diag( np.ones( G.shape[ 0 ] ) ) ##weighted degrees of the nodes in G
		for i in range( D.shape[ 0 ] ):
			D[ i , i ] = np.sum( G[ i , : ] )

		combinatorial_laplacian = D - G

		return combinatorial_laplacian

###################################################################HELPERS##################################################################################

