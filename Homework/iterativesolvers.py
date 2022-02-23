"""iterativesolvers.py: Implements the Jacobi and Gauss-Siedel Iterative Solvers."""

__author__      = "Tony Saad"
__copyright__   = "Copyright 2018, Tony Saad under the MIT license"

import numpy as np
from numpy.linalg import norm

def jacobi(Amat,xguesst,rhs,tol,maxIter,inorm=2):
	'''
	Amat: Coefficient matrix (size nxn)
	xguesst: Vector of size n containing initial guesses
	rhs: Vector of size n containing the right-hand-side of the system of equations
	tol: Tolerance
	maxIter: Maximum number of iterations to take, even if tolerance has not been reached
	inorm: Vector norm to be used. Defaults to L2 norm
	'''
	A, xguess, b = map(np.array, (Amat, xguesst, rhs))     # copy the arrays
	# make sure the matrix dimensions are the same
	[nr,nc]=A.shape
	if (nr != nc or nr != len(b) or len(b) != len(xguess)):
			print("Error! Dimensions do not match!")
			return
	# define the norm:
	r = b - A.dot(xguess)
	res = norm(r,np.inf)
	iters=0 # counter
	x = np.zeros(nr) # declare x array
	d  = A.diagonal()
	while ( (abs(res) > tol) and (iters <= maxIter) ): 
			# loop over the rows
			for row in range(0,nr):
					x[row] = xguess[row] + 1.0/A[row][row] * (b[row] - A[row].dot(xguess))                    
			r = b - A.dot(x)
			xguess = x.copy() # this is important! make sure x and xguess are NOT the same
			res = norm(r,inorm)
			iters +=1
	return x,res,iters
		
def gauss_siedel(Amat,xguesst,rhs,tol,maxIter, inorm=2):
	'''
	Amat: Coefficient matrix (size nxn)
	xguesst: Vector of size n containing initial guesses
	rhs: Vector of size n containing the right-hand-side of the system of equations
	tol: Tolerance
	maxIter: Maximum number of iterations to take, even if tolerance has not been reached
	inorm: Vector norm to be used. Defaults to L2 norm
	'''
	A, xguess, b = map(np.array, (Amat, xguesst, rhs))     # copy the arrays
	# make sure the matrix dimensions are the same
	nr = A.shape[0]
	nc = A.shape[1]
	d  = A.diagonal()
	if (nr != nc):
			print("Error! Matrix dimensions do not match!")
			return
	# define the norm:
	r = b - A.dot(xguess)
	res = np.linalg.norm(r,np.inf)
	iters=0
	x = np.array(xguess)
	while ( (abs(res) > tol) and (iters <= maxIter) ): 

			# loop over the rows
			for row in range(0,nr):
					x[row] = x[row] + 1.0/A[row][row] * (b[row] - A[row].dot(x))                    

			r = b - A.dot(x)

			res = np.linalg.norm(r,inorm)
			iters +=1

	return x,res,iters