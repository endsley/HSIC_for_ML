#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append('./src')
sys.path.append('./src/helper')
from kernelNet import *
from classifier import *
from sklearn.metrics import accuracy_score


##	Example 1 ,  clustering wine dataset 
db = {}
db['X'] = np.loadtxt('datasets/wine_75.00_validation.csv', delimiter=',', dtype=np.float64)			
db['Y'] = np.loadtxt('datasets/wine_75.00_label_validation.csv', delimiter=',', dtype=np.int32)			
db['num_of_clusters'] = 3
db['print_debug'] = True

kn = kernelNet(db)
[Ψx, U, U_normalized] = kn.train()

label = np.loadtxt('datasets/wine_75.00_label_validation.csv', delimiter=',', dtype=np.int32)			
[allocation, Ψx_nmi] = kmeans(db['num_of_clusters'], Ψx, Y=label)
[allocation, U_nmi] = kmeans(db['num_of_clusters'], U, Y=label)
[allocation, U_normalized_nmi] = kmeans(db['num_of_clusters'], U_normalized, Y=label)

print('NMI Ψx_nmi : %.3f'%(Ψx_nmi))
print('NMI U : %.3f'%(U_nmi))
print('NMI U_normalized : %.3f'%(U_normalized_nmi))




##	Example 2 ,  Clustering cancer data
#db = {}
#db['X'] = np.loadtxt('datasets/breast_30.00_validation.csv', delimiter=',', dtype=np.float64)			
#db['Y'] = np.loadtxt('datasets/breast_30.00_label_validation.csv', delimiter=',', dtype=np.int32)			
#db['num_of_clusters'] = 2
#db['print_debug'] = True
#
#kn = kernelNet(db)
#[Ψx, U, U_normalized] = kn.train()
#
#label = np.loadtxt('datasets/breast_30.00_label_validation.csv', delimiter=',', dtype=np.int32)			
#[allocation, Ψx_nmi] = kmeans(db['num_of_clusters'], Ψx, Y=label)
#[allocation, U_nmi] = kmeans(db['num_of_clusters'], U, Y=label)
#[allocation, U_normalized_nmi] = kmeans(db['num_of_clusters'], U_normalized, Y=label)
#
#print('NMI Ψx_nmi : %.3f'%(Ψx_nmi))
#print('NMI U : %.3f'%(U_nmi))
#print('NMI U_normalized : %.3f'%(U_normalized_nmi))





##	Example 3 ,  Clustering spiral
#db = {}
#db['X'] = np.loadtxt('datasets/spiral.csv', delimiter=',', dtype=np.float64)			
#db['Y'] = np.loadtxt('datasets/spiral_label.csv', delimiter=',', dtype=np.int32)			
#db['num_of_clusters'] = 3
#db['width_scale'] = 6
#db['σ_ratio'] = 0.1
#db['print_debug'] = True
#
#
#kn = kernelNet(db)
#[Ψx, U, U_normalized] = kn.train()
#
#
#label = np.loadtxt('datasets/spiral_label.csv', delimiter=',', dtype=np.int32)			
#[allocation, Ψx_nmi] = kmeans(db['num_of_clusters'], Ψx, Y=label)
#[allocation, U_nmi] = kmeans(db['num_of_clusters'], U, Y=label)
#[allocation, U_normalized_nmi] = kmeans(db['num_of_clusters'], U_normalized, Y=label)
#
#print('NMI Ψx_nmi : %.3f'%(Ψx_nmi))
#print('NMI U : %.3f'%(U_nmi))
#print('NMI U_normalized : %.3f'%(U_normalized_nmi))



