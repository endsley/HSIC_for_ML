#!/usr/bin/env python

from kernel_lib import *


def add_rank_constraint(db, Φ):
	if db['λ_ratio'] == 0: return Φ

	A = db['W'].dot(db['W'].T) + 0.001*np.eye(db['data'].d)
	rc = np.linalg.inv(A)

	if 'λ' not in db:
		Φ_norm = np.linalg.norm(Φ)
		rc = np.linalg.norm(rc)
		db['λ'] = db['λ_ratio']*Φ_norm/rc

	#Φ_norm = np.linalg.norm(Φ)
	#rc = np.linalg.norm(rc)
	#print('%.3f + %.3f * %.3f'%(Φ_norm, rc, db['λ']))
	return Φ + db['λ']*rc


def relative_Φ_0(db):
	X = db['data'].X
	σ = db['data'].σ

	γ = db['compute_γ']()
	Ψ = db['Σ']*γ

	D_Ψ = compute_Degree_matrix(Ψ)
	Φ = X.T.dot(D_Ψ - Ψ).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)
	Φ = add_rank_constraint(db, Φ)

	return Φ

def learn_center_alignment_weights(db):
	X = db['data'].X
	σ = db['data'].σ

	Kg = rbk_sklearn(X, σ)
	Kg = center_scale_entire_kernel(Kg)
	HKgH = double_center(Kg, db['H'])
	HKuH = db['compute_γ']()
	gaussian_μ = np.sum(Kg*HKuH)/np.sqrt(np.sum(Kg*HKgH)*np.sum(db['Ku']*HKuH))

	Kp = poly_sklearn(X, db['poly_power'], db['poly_constant'])
	Kp = center_scale_entire_kernel(Kp)
	HKpH = double_center(Kp, db['H'])
	poly_μ = np.sum(Kp*HKuH)/np.sqrt(np.sum(Kp*HKpH)*np.sum(db['Ku']*HKuH))

	#Kl = X.dot(X.T)
	#Kl = center_scale_entire_kernel(Kl)
	#HKlH = double_center(Kl, db['H'])
	#linear_μ = np.sum(Kl*HKuH)/np.sqrt(np.sum(Kl*HKpH)*np.sum(db['Ku']*HKuH))

	#db['gaussian_μ'] = gaussian_μ/(gaussian_μ + poly_μ + linear_μ)
	#db['poly_μ'] = poly_μ/(gaussian_μ + poly_μ + linear_μ)
	#db['linear_μ'] = linear_μ/(gaussian_μ + poly_μ + linear_μ)

	db['gaussian_μ'] = gaussian_μ/(gaussian_μ + poly_μ)
	db['poly_μ'] = poly_μ/(gaussian_μ + poly_μ)

	#db['gaussian_μ'] = 0.60
	#db['poly_μ'] = 1 - db['gaussian_μ']

def mkl_Φ_0(db):
	learn_center_alignment_weights(db)

	Φg = gaussian_Φ_0(db)
	Φp = polynomial_Φ_0(db)

	#Φg = center_scale_entire_kernel(gaussian_Φ_0(db))
	#Φp = center_scale_entire_kernel(polynomial_Φ_0(db))

	Φ = db['gaussian_μ']*Φg + db['poly_μ']*Φp #+ db['linear_μ']*linear_Φ_0(db)
	return Φ


def multiquadratic_Φ_0(db):
	X = db['data'].X

	γ = db['compute_γ']()
	D_γ = compute_Degree_matrix(γ)
	Φ = X.T.dot(D_γ - γ).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)
	Φ = add_rank_constraint(db, Φ)

	return Φ



def squared_Φ_0(db):
	X = db['data'].X

	γ = db['compute_γ']()
	D_γ = compute_Degree_matrix(γ)
	Φ = X.T.dot(D_γ - γ).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)
	Φ = add_rank_constraint(db, Φ)

	return Φ


def gaussian_Φ_0(db):
	X = db['data'].X
	σ = db['data'].σ

	γ = db['compute_γ']()
	D_γ = compute_Degree_matrix(γ)

	Φ = X.T.dot(D_γ - γ).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)
	Φ = add_rank_constraint(db, Φ)

	return Φ

def polynomial_Φ_0(db):
	X = db['data'].X
	p = db['poly_power']
	c = db['poly_constant']

	γ = db['compute_γ']()
	Φ = -X.T.dot(γ).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)
	Φ = add_rank_constraint(db, Φ)

	return Φ

def linear_Φ_0(db):
	X = db['data'].X
	γ = db['compute_γ']()
	Φ = -X.T.dot(γ).dot(X); 			
	Φ = add_rank_constraint(db, Φ)

	return Φ

def gaussian_Φ_slow(db):
	X = db['data'].X
	σ = db['data'].σ
	γ = db['compute_γ']()

	[Kx, D] = Kx_D_given_W(db)
	Ψ=γ*Kx

	Φ = np.zeros((db['data'].d, db['data'].d))
	for m in range(db['data'].N):	
		for n in range(db['data'].N):	
			ΔX = X[m,:] - X[n,:]
			A_ij = np.outer(ΔX,ΔX)
			Φ = Φ + Ψ[m,n]*A_ij

	Φ = add_rank_constraint(db, Φ)
	return Φ

def multiquadratic_Φ(db):
	X = db['data'].X

	Kx = 1.0/multiquadratic_kernel(X)
	γ = db['compute_γ']()
	Ψ=γ*Kx
	D_Ψ = compute_Degree_matrix(Ψ)
	Φ = -X.T.dot(D_Ψ - Ψ).dot(X) 			#debug.compare_Φ(db, Φ, Ψ)
	Φ = add_rank_constraint(db, Φ)

	return Φ



def gaussian_Φ(db):
	X = db['data'].X
	σ = db['data'].σ

	γ = db['compute_γ']()

	[Kx, D] = Kx_D_given_W(db)
	Ψ=γ*Kx
	D_Ψ = compute_Degree_matrix(Ψ)
	Φ = X.T.dot(D_Ψ - Ψ).dot(X) 			#debug.compare_Φ(db, Φ, Ψ)
	Φ = add_rank_constraint(db, Φ)

	return Φ

def relative_Φ(db):
	X = db['data'].X
	σ = db['data'].σ

	γ = db['compute_γ']()

	[Kx, D] = Kx_D_given_W(db)
	Ψ=db['Σ']*γ*Kx
	D_Ψ = compute_Degree_matrix(Ψ)
	Φ = X.T.dot(D_Ψ - Ψ).dot(X) 			#debug.compare_Φ(db, Φ, Ψ)
	Φ = add_rank_constraint(db, Φ)

	return Φ

def polynomial_Φ(db):
	X = db['data'].X
	p = db['poly_power']
	c = db['poly_constant']


	Kx = poly_sklearn(X.dot(db['W']), p-1, c)
	γ = db['compute_γ']()
	Ψ = γ*Kx
	Φ = -X.T.dot(Ψ).dot(X); 			#debug.compare_Φ(db, Φ, Ψ)
	Φ = add_rank_constraint(db, Φ)

	return Φ

def linear_Φ(db):
	X = db['data'].X
	γ = db['compute_γ']()
	Φ = -X.T.dot(γ).dot(X); 			
	Φ = add_rank_constraint(db, Φ)

	return Φ

def mkl_Φ(db):
	#Φ = db['gaussian_μ']*gaussian_Φ(db) + db['poly_μ']*polynomial_Φ(db) + db['linear_μ']*linear_Φ(db)

	#Φg = center_scale_entire_kernel(gaussian_Φ_0(db))
	#Φp = center_scale_entire_kernel(polynomial_Φ_0(db))
	Φg = gaussian_Φ_0(db)
	Φp = polynomial_Φ_0(db)

	Φ = db['gaussian_μ']*Φg + db['poly_μ']*Φp
	Φ = add_rank_constraint(db, Φ)

	return Φ
