#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append('./src')
from LUDR import *
from sklearn.metrics import accuracy_score


db = {}
db['X'] = np.loadtxt('datasets/breast_30.00_validation.csv', delimiter=',', dtype=np.float64)			
db['Y'] = np.loadtxt('datasets/breast_30.00_label_validation.csv', delimiter=',', dtype=np.int32)			
db['num_of_clusters'] = 2
db['q'] = 4


dr_c = LUDR(db)
dr_c.train()

W = dr_c.get_projection_matrix()
new_X = dr_c.get_reduced_dim_data()
allocation = dr_c.get_clustering_result()
nmi = normalized_mutual_info_score(allocation, db['Y'])

print('Original dimension : %d X %d'%(db['X'].shape[0], db['X'].shape[1]))
print('Reduced dimension : %d X %d'%(new_X.shape[0], new_X.shape[1]))
print('Clustering quality in NMI : %.3f'%(nmi))





