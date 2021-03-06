

from matplotlib import pyplot as plt
import numpy as np

def plot_alloc(db, plotID, data, title, linetype=None, fsize=20, xyLabels=[]):
	color_list = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']

	plt.subplot(plotID)
	Uq_a = np.unique(db['allocation'])
	
	for m in range(len(Uq_a)):
		g = data[db['allocation'] == Uq_a[m]]
		if linetype is None:
			plt.plot(g[:,0], g[:,1], color_list[m] + 'o')
		else:
			plt.plot(g[:,0], g[:,1], color_list[m] + linetype[m])

	#plt.tick_params(labelsize=9)
	plt.title(title, fontsize=fsize, fontweight='bold')
	if len(xyLabels) > 1:
		plt.xlabel(xyLabels[0], fontsize=fsize, fontweight='bold')
		plt.ylabel(xyLabels[1], fontsize=fsize, fontweight='bold')
	plt.tick_params(labelsize=10)

def plot_output(db):
	if 'running_batch_mode' in db: return

	[current_loss, current_hsic, current_AE_loss, φ_x, U, U_normalized] = db['knet'].get_current_state(db, db['train_data'].X_Var)
	db['allocation'] = kmeans(db['num_of_clusters'], U_normalized)
	#plt.plot(φ_x[:,0], φ_x[:,1], 'go')
	plot_alloc(db, 111, db['train_data'].X, '', linetype=None, fsize=20, xyLabels=[])

	plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
	plt.show()

	import pdb; pdb.set_trace()

