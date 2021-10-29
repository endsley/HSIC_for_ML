#!/usr/bin/env python3

from algorithm import *
from kernel_lib import *
from terminal_print import *
from gradients import *
from Φs import *
from math import e
from sklearn.cluster import SpectralClustering
#import debug

#	max Tr[(DKuD)Kx]
#	W -> Kx -> D -> γ -> Σ ψA_i,j -> W
class linear_unsupv_dim_reduction(algorithm):
	def __init__(self, db):
		db['W'] = np.zeros((db['data'].d ,db['q']))
		#self.Ku = Ku_kernel(db['data'].Y)
		N = db['data'].N
		db['H'] = np.eye(N) - (1.0/N)*np.ones((N,N))

		if db['W_optimize_technique'].__name__ == 'grassman':
			db['compute_cost'] = self.grassman_cost_function
		else: db['compute_cost'] = self.compute_cost	

		db['compute_gradient'] = self.compute_f_gradient
		db['compute_Φ'] = self.compute_Φ
		db['compute_γ'] = self.update_γ	

		self.λ0 = None
		self.conv_threshold = 0.01
		self.W = None
		self.W_λ = None
		self.U_λ = None

		algorithm.__init__(self, db)
		print('Experiment : linear unsupervised dimensionality reduction\n')

	def grassman_cost_function(self, W):
		new_X = np.dot(self.db['data'].X, W)
		σ = self.db['data'].σ
		γ = self.db['compute_γ']()

		#	compute gaussian kernel
		bs = new_X.shape[0]
		K = np.empty((0, bs))	
		for i in range(bs):
			Δx = new_X[i,:] - new_X
			exp_val = -np.sum(Δx*Δx, axis=1)/(2*σ*σ)
			K = np.vstack((K, e**(exp_val)))

		return -np.sum(γ*K)

	def update_γ(self):
		db = self.db
		Ku = self.U.dot(self.U.T)
		γ = double_center(Ku, db['H'])
		return γ

		#DγD = db['D_inv'].dot(γ).dot(db['D_inv'])
		#return DγD

		#return γ

	def compute_f_gradient(self, old_x):
		self.db['W'] = old_x
		return compute_objective_gradient(self.db)

	def compute_Φ(self, old_x):
		self.db['W'] = old_x

		if self.db['kernel_type'] == 'rbf':
			return gaussian_Φ(self.db)
		elif self.db['kernel_type'] == 'rbf_slow':
			return gaussian_Φ_slow(self.db)
		elif self.db['kernel_type'] == 'linear':
			return linear_Φ(self.db)
		elif self.db['kernel_type'] == 'polynomial':
			return polynomial_Φ(self.db)
		elif self.db['kernel_type'] == 'relative':
			return relative_Φ(self.db)
		elif self.db['kernel_type'] == 'squared':
			return squared_Φ_0(self.db)
		elif self.db['kernel_type'] == 'multiquadratic':
			return multiquadratic_Φ(self.db)
		elif self.db['kernel_type'] == 'mkl':	# multiple kernel learning
			return mkl_Φ(self.db)
		else:	
			print('\n\nError : in unsupv_dim_reduction unrecognized kernel type : %s'%self.db['kernel_type'])
			sys.exit()

	def update_f(self):
		db = self.db
		#write_to_current_line('\tAt update_f\n')
		self.db['W'] = self.optimizer.run(self.db['W'])

		### debug
		#nmi = self.alt_Spectral_Clustering(db)
		#print('\t\tupdate f NMI %.3f'%nmi)
		#import pdb; pdb.set_trace()	



	def compute_cost(self, W=None):
		[Kx, D] = Kx_D_given_W(self.db, setW=W)
		γ = self.update_γ()					#DHKuHD
		return -np.sum(γ*Kx)

	def update_U(self):
		db = self.db
		k = self.db['num_of_clusters']

		[Kx, db['D_inv']] = Kx_D_given_W(db, setW=db['W'])
		#if self.db['kernel_type'] == 'rbf': L = db['D_inv'].dot(Kx).dot(db['D_inv'])
		#else: L = Kx
		L = double_center(Kx, db['H'])

		[self.U, U_λ] = eig_solver(L, k, mode='largest')

		if self.U_λ is None:
			self.U_diff = 1
		else:
			self.U_diff = np.linalg.norm(self.U_λ - U_λ)/np.linalg.norm(self.U_λ)
	
		self.U_λ = U_λ
		#print(self.U[0:3,:])
		#print('\n\nUpdate U')
		#import pdb; pdb.set_trace()	

	def initialize_U(self):
		db = self.db
		k = db['num_of_clusters']
		X = db['data'].X
		σ = db['data'].σ
		default_to_relative_initial_kernel = True

		if default_to_relative_initial_kernel:
			σ_list = 1.0/relative_σ(db['data'].X)
			db['Σ'] = σ_list.dot(σ_list.T)
			Kx = rbk_relative_σ(db, db['data'].X)
		else:
			if self.db['kernel_type'] == 'relative':
				σ_list = 1.0/relative_σ(db['data'].X)
				db['Σ'] = σ_list.dot(σ_list.T)
			elif self.db['kernel_type'] == 'mkl':	# multiple kernel learning
				print('\n\nError : mkl kernel type is for supervised only...')
				sys.exit()
	
			[Kx, db['D_inv']] = Kx_D_given_W(db, setW=np.eye(db['data'].d))
			#L = db['D_inv'].dot(Kx).dot(db['D_inv'])


		L = double_center(Kx, db['H'])
		[self.U, self.U_λ] = eig_solver(L, k, mode='largest')

		U_normed = normalize_U(self.U)
		[allocation, self.original_nmi] = kmeans(k, U_normed, db['data'].Y)

		#print('\t\tOriginal NMI %.3f'%self.original_nmi)
		#import pdb; pdb.set_trace()	

	def initialize_W(self):
		db = self.db

		if db['kernel_type'] == 'rbf':
			Φ = gaussian_Φ_0(db)
		elif db['kernel_type'] == 'rbf_slow':
			Φ = gaussian_Φ_0(db)
		elif db['kernel_type'] == 'linear':
			Φ = linear_Φ_0(db)
		elif db['kernel_type'] == 'polynomial':
			Φ = polynomial_Φ_0(db)
		elif db['kernel_type'] == 'relative':
			if 'Σ' not in db:
				σ_list = 1.0/relative_σ(db['data'].X)
				db['Σ'] = σ_list.dot(σ_list.T)
			Φ = relative_Φ_0(db)
		elif self.db['kernel_type'] == 'squared':
			Φ = squared_Φ_0(self.db)
		elif self.db['kernel_type'] == 'multiquadratic':
			Φ = multiquadratic_Φ_0(self.db)
		elif self.db['kernel_type'] == 'mkl':	# multiple kernel learning
			Φ = mkl_Φ_0(self.db)
		else:	
			print('\n\nError : unrecognized kernel type : %s'%self.db['kernel_type'])
			sys.exit()


		[db['W'], db['W_λ']] = eig_solver(Φ, db['q'], mode='smallest')
	

		#### debug
		#nmi = self.alt_Spectral_Clustering(db)
		#print('\t\tOriginal W NMI %.3f'%nmi)
		##import pdb; pdb.set_trace()	


	def get_clustering_result(self):
		Y = self.db['data'].Y
		k = self.db['num_of_clusters']
		
		U_normed = normalize_U(self.U)
		allocation = kmeans(k, U_normed)
		return allocation

	def outer_converge(self):
		if self.U_diff < 0.01:
			#print('\tU_diff %.3f'% self.U_diff)
			return True
		else:
			#print('\tU_diff %.3f'% self.U_diff)
			return False

	def alt_Spectral_Clustering(self, db):
		k = db['num_of_clusters']
		[Kx, db['D_inv']] = Kx_D_given_W(db, setW=db['W'])
		#L = db['D_inv'].dot(Kx).dot(db['D_inv'])
		L = Kx

		[self.U, self.U_λ] = eig_solver(L, k, mode='largest')
	
		U_normed = normalize_U(self.U)
		[allocation, nmi] = kmeans(k, U_normed, db['data'].Y)
		return nmi


	def verify_result(self, start_time):
		db = self.db
		if 'ignore_verification' in db: return

		k = db['num_of_clusters']
		σ = db['data'].σ

		final_cost = self.compute_cost()
		db['data'].load_validation()
		outstr = '\nExperiment : linear unsupervised dimensionality reduction : %s with final cost : %.3f\n'%(db['data_name'], final_cost)

		Y = db['data'].Y
		X = db['data'].X

		X_valid = db['data'].X_valid
		Y_valid = db['data'].Y_valid

		outstr += self.verification_basic_info(start_time)

		nmi = self.alt_Spectral_Clustering(db)
		#alloc = SpectralClustering(k, gamma=1/(2*σ*σ)).fit_predict(X.dot(db['W']))
		#nmi = normalized_mutual_info_score(alloc, Y)

		outstr += '\t\tTraining clustering NMI without dimension reduction : %.3f'%self.original_nmi + '\n'
		outstr += '\t\tTraining clustering NMI with dimension reduction : %.3f'%nmi + '\n'


		if db['separate_data_for_validation']:
			[allocation, nmi_orig] = my_spectral_clustering(X_valid, k, σ, H=db['H'], Y=Y_valid)
			nmi = self.alt_Spectral_Clustering(db)

			outstr += '\t\tTest clustering NMI without dimension reduction : %.3f'%nmi_orig + '\n'
			outstr += '\t\tTest clustering NMI with dimension reduction : %.3f'%nmi + '\n'



		start_time = time.time() 
		pca = PCA(n_components=db['q'])
		X_pca1 = pca.fit_transform(X)
		Xpca = pca.transform(X_valid)
		[allocation, pca_nmi] = my_spectral_clustering(Xpca, k, σ, H=self.db['H'], Y=Y_valid)
		pca_time = time.time() - start_time
		outstr += '\tPCA\n'
		outstr += '\t\tTraining Clustering NMI with PCA dimension reduction : %.3f'%pca_nmi + '\n'
		outstr += '\t\trun time : %.3f'%pca_time + '\n'



		print(outstr)

		fin = open('./results/LUDR_' + db['data_name'] + '_' + db['kernel_type'] + '_' + db['W_optimize_technique'].__name__ + '.txt', 'w') 
		fin.write(outstr)
		fin.close()

