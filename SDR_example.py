#!/usr/bin/env python3

import sys
import numpy as np
sys.path.append('./src')
from LSDR import *
from sklearn.metrics import accuracy_score



db = {}
db['num_of_clusters'] = 26
db['q'] = 26
db['λ_ratio'] = 0.0
original_name = db['data_name'] = 'raman' 	# raman, wine
train_acc_list = []
test_acc_list = []
#ratios = np.arange(0,2,0.05)
ratios = [200]

for m in ratios:
	db['λ_ratio'] = m
	for i in range(1, 11):
		db['data_file_name'] = './datasets/' + original_name + '/' + original_name + '_' + str(i) + '.csv'
		db['label_file_name'] = './datasets/' + original_name + '/' + original_name + '_' + str(i) + '_label.csv'
	
		db['validation_data_file_name'] = './datasets/' + original_name + '/' + original_name + '_' + str(i) + '_validation.csv'
		db['validation_label_file_name'] = './datasets/' + original_name + '/' + original_name + '_' + str(i) + '_label_validation.csv'
	
		print('loading : %s'%db['data_file_name'])
	
		db['X'] = np.loadtxt(db['data_file_name'], delimiter=',', dtype=np.float64)			
		db['Y'] = np.loadtxt(db['label_file_name'], delimiter=',', dtype=np.int32)			
		db['X2'] = np.loadtxt(db['validation_data_file_name'], delimiter=',', dtype=np.float64)			
		db['Y2'] = np.loadtxt(db['validation_label_file_name'], delimiter=',', dtype=np.int32)			
	
		db['X'] = preprocessing.scale(db['X'])
		db['X2'] = preprocessing.scale(db['X2'])
	
	
		sdr = LSDR(db)
		sdr.train()
	
		W = sdr.get_projection_matrix()
		new_X = sdr.get_reduced_dim_data()
	
	
		[out_allocation, nmi, svm_time, svm_object] = use_svm(db['X'], db['Y'], W)
		acc_train = accuracy_score(db['Y'], out_allocation)
		train_acc_list.append(acc_train)
	
		[out_allocation2, nmi_2, svm_time_2] = predict_with_svm(svm_object, db['X2'], db['Y2'], W)
		acc_test = accuracy_score(db['Y2'], out_allocation2)
		test_acc_list.append(acc_test)
	
		#print('\tOriginal dimension : %d X %d'%(db['X'].shape[0], db['X'].shape[1]))
		#print('\tReduced dimension : %d X %d'%(new_X.shape[0], new_X.shape[1]))
		#print('\tClassification quality in Training Accuracy : %.3f'%(acc_train))
		#print('\tClassification quality in Test Accuracy : %.3f'%(acc_test))
		#print('\tClassification quality in Training NMI : %.3f'%(nmi))
		#print('\tClassification quality in Test NMI : %.3f'%(nmi_2))
	
	outstr = 'λ_ratio : %.2f\n'%db['λ_ratio']
	#outstr += '\ttrain accuracy list : %s\n'%str(train_acc_list)
	#outstr += '\ttest accuracy list : %s'%str(test_acc_list)
	outstr += '\tmean train accuracy : %.3f ± %.3f\n'%(np.mean(train_acc_list), np.std(train_acc_list))
	outstr += '\tmean test accuracy : %.3f ± %.3f\n\n'%(np.mean(test_acc_list), np.std(test_acc_list))
	print(outstr)

	fin = open('results/lambda_ratio.txt','a') 
	fin.write(outstr)
	fin.close()
