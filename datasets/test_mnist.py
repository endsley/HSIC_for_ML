#!/usr/bin/env python


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score



dtype = np.float64				#np.float32
X = np.loadtxt('mnist.csv', delimiter=',', dtype=dtype)			
Y = np.loadtxt('mnist_label.csv', delimiter=',', dtype=dtype)			

allocation = KMeans(10).fit_predict(X)
nmi = normalized_mutual_info_score(allocation, Y)

print(nmi)
