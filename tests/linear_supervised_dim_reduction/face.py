#!/usr/bin/env python3

from test_base import *
import sklearn.metrics
import numpy as np
from subprocess import call
from basic_dataset import *
from linear_supv_dim_reduction import *
from ism import *
from orthogonal_optimization import *
from DimGrowth import *
from grassman import *

class test_obj(test_base):
	def __init__(self):
		db = {}
		db['data_name'] = 'face'
		db['data_source'] = 'numpy_files'				# link_download, load_image, local_file

		db['dataset_class'] = basic_dataset
		db['TF_obj'] = linear_supv_dim_reduction
		db['W_optimize_technique'] = ism  			# orthogonal_optimization, ism, DimGrowth, grassman
		db['compute_error'] = None
		db['store_results'] = None
		db['separate_data_for_validation'] = False

		db['q'] = 20
		db['num_of_clusters'] = 20
		db['σ_ratio'] = 1.0							# multiplied to the median of pairwise distance as sigma
		db['λ_ratio'] = 0.0							# rank constraint ratio
		db['kernel_type'] = 'mkl'					# linear, mkl, rbf, polynomial

		test_base.__init__(self, db)


	def parameter_ranges(self):
		#W_optimize_technique = [ism, DimGrowth, orthogonal_optimization, grassman]	
		W_optimize_technique = [ism]	
		kernel_type = ['linear', 'squared','multiquadratic']
	
		return [W_optimize_technique, kernel_type]

	def run_10_fold_single(self, indx_list):
		db = self.db

		db['W_optimize_technique'] = DimGrowth
		db['separate_data_for_validation'] = True

		for indx in indx_list:
			self.kick_off_single_from_10_fold(indx)


	def run_10_fold(self):
		db = self.db
		#W_optimize_technique = [ism, orthogonal_optimization, DimGrowth, grassman]		
		W_optimize_technique = [ism]		
		db['separate_data_for_validation'] = True
		self.gen_10_fold_data()

		for technique in W_optimize_technique:
			db['W_optimize_technique'] = technique
			self.kick_off_each()

	def gather_10_fold_result(self):
		#W_optimize_technique = [ism, DimGrowth, orthogonal_optimization, grassman]	
		W_optimize_technique = [ism]	
		db = self.db

		for technique in W_optimize_technique:
			print('Running %s '%technique.__name__)
			db['W_optimize_technique'] = technique
			self.collect_10_fold_info(technique)


prog = test_obj()
#prog.run_10_fold_single([8,9,10])
prog.run_10_fold()
prog.gather_10_fold_result()
#prog.collect_10_fold_info(ism)
#prog.basic_run()
#prog.batch_run()

