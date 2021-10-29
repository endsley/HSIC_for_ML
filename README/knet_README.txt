
Kernel Net by Chieh Wu
Last Update : Feb/19/19

This code is the continual development of the original Kernel Net. We have discovered 
that instead of computing the HSIC objective stochastically, it appears to work better
if we compute the entire kernel. However due to the computational complexity, we 
approximate the entire kernel using the Random Fourier Feature. 

The 2nd change is that instead of pre-training the autoencoder so that Ψ(X) = X, we 
use a predefined identity network where the input and the output are automatically
eqaul. Therefore, no pre-training is required. A separate paper might be required
to explain the construct of this identity network.

The 3rd change is the addition of the sparcity L1 contraint. For the objective, we
added the L1 norm for |Ψ(X)|. It allows use to not only discover a convex representation
but a sparse and convex representation. The constant for the regularizer is set as
the ratio between the original HSIC objective and L1 norm multiplied by 0.6, i.e., 
(0.6) HSIC/|Ψ(X)|. 


Future work : 
	1. Instead of the autoencoder, we still need to look into using 
	HSIC(X,|Ψ(X)|) as another regularizer. This constraint forces |Ψ(X)| to be dependent
	on X.
	2. Unfortunately, the new changes renders the code significantly slower. More 
	work is require to speed up the computation.
