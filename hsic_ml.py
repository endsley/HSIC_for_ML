#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import sys
import matplotlib
import numpy as np
import random
import itertools
import socket


sys.path.append('./src')
sys.path.append('./src/data_loader')
sys.path.append('./src/algorithms')
sys.path.append('./src/helper')
sys.path.append('./src/optimization')
sys.path.append('./src/networks')
sys.path.append('./tests')


#sys.path.append('./models')
#sys.path.append('./optimizer')
#sys.path.append('./training_models')
#sys.path.append('../img_manipulate')


if socket.gethostname() == 'discovery4.neu.edu':
	print('\nError : you cannot run program on discovery4.neu.edu.......\n\n')
	sys.exit()



#	supervised linear dim reduction
sys.path.append('./tests/linear_supervised_dim_reduction')
from spiral import *									
#from wine_75 import *									
#from breast_30 import *									
#from car import *									
#from face import *									
#from mnist import *									
#from raman import *									


#sys.path.append('./tests/linear_supervised_dim_reduction')
#from face_20 import *									
#from face_40 import *									
#from face_60 import *									
#from face_80 import *									
#from face import *									



####	unsupervised linear dim reduction
#sys.path.append('./tests/linear_unsupervised_dim_reduction')
#from wine_75 import *									
#from breast_30 import *									
#from car import *									
#from face import *									
#from mnist import *									

###	sknet
#sys.path.append('./tests/sknet'); 
#from wine import *									
#from cancer import *									


##	Knet
#sys.path.append('./tests/knet'); 
#from wine import *									
#from cancer import *									
#from face import *									

