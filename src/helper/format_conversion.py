#!/usr/bin/env python3

import numpy as np
import torch
from torch.autograd import Variable

def ensure_matrix_is_numpy(U):
	if type(U) == torch.DoubleTensor:
		U = U.numpy()
	elif type(U) == np.ndarray:
		pass
	elif type(U) == torch.Tensor:
		if U.device.type == 'cuda':
			U = U.cpu().data.numpy()	
		elif U.device.type == 'cpu':
			U = U.data.numpy()
	elif type(U) == torch.cuda.FloatTensor:
		U = U.data.cpu().numpy()
	elif type(U) == torch.FloatTensor:
		U = U.numpy()
#	elif type(U) == torch.autograd.variable.Variable:
#		U = U.data.numpy()
	elif type(U) == np.numpy_boxes.ArrayBox:
		U = U._value
	else:
		raise
	return U


#def ensure_matrix_is_numpy(U):
#	if type(U) == torch.DoubleTensor:
#		U = U.numpy()
#	elif type(U) == np.ndarray:
#		pass
#	elif type(U) == torch.FloatTensor:
#		U = U.numpy()
#	elif type(U) == torch.autograd.variable.Variable:
#		U = U.data.numpy()
#	else:
#		raise
#	return U
#

