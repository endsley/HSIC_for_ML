#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


#for i in range(1, 10):
#	n = i*100
#	d = 2
#	
#	X1 = np.random.randn(n,d) + np.array([3,3])
#	X2 = np.random.randn(n,d) + np.array([-3,-3])
#	X = np.vstack((X1,X2))
#	
#	Y1 = np.ones((n,1))
#	Y2 = np.zeros((n,1))
#	Y = np.vstack((Y1,Y2))
#	
#	np.savetxt('gauss_' + str(n*2) + '_validation.csv', X, delimiter=',', fmt='%3f') 
#	np.savetxt('gauss_' + str(n*2) + '_label_validation.csv', Y, delimiter=',', fmt='%d')




for i in range(1, 10):
	X1 = np.random.randn(200,2) + np.array([3,3])
	X2 = np.random.randn(200,2) + np.array([-3,-3])
	X = np.vstack((X1,X2))

	d = 20 * i
	noise = 0.1*np.random.randn(400,d)
	X = np.hstack((X, noise))

	Y1 = np.ones((200,1))
	Y2 = np.zeros((200,1))
	Y = np.vstack((Y1,Y2))
	
	np.savetxt('gaussD_' + str(d+2) + '_validation.csv', X, delimiter=',', fmt='%3f') 
	np.savetxt('gaussD_' + str(d+2) + '_label_validation.csv', Y, delimiter=',', fmt='%d')



#area = 2
#plt.scatter(X[:,0], X[:,1], s=1, alpha=0.5)
#plt.show()

