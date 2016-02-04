from numpy import *
from scipy import linalg,array,dot,mat
from scipy.sparse.linalg import eigsh
from math import *
from pprint import pprint

def createLaplacian(DoctermMatrix):
	# Create word-word mapping matrix
	# WW and DD to be eventually constructed from an external source
	# Created from sum of word and docs (as per example in Hendrickson paper)
	#WW  = diag(add.reduce(DoctermMatrix),0) 
	#DD = diag(add.reduce(DoctermMatrix, axis=1),0) 
	
	# Create word-word mapping matrix
	# Initially constructed from the doc-word matrix	
	WW = dot(DoctermMatrix.transpose(),DoctermMatrix) * (-1)
	# Create doc-doc mapping matrix	
	DD = dot(DoctermMatrix,DoctermMatrix.transpose()) * (-1)
	
	B = DoctermMatrix * (-1)
	BT = B.transpose()
	
	# Create Block Matrix L
	# L is a (nodocs + noterms) by (nodocs + noterms) matrix
	#  ---      ---
	# WW    BT  
	# B        DD
	#  ---      ---
	L = bmat('WW,BT; B,DD')

	# Make sure the diagonal values make the row-sums add to zero
	#Set the diagonals of L to 0
	fill_diagonal(L, 0)
	L_diag = L.sum(axis=1) * (-1)
	fill_diagonal(L, L_diag)
	return L


def fiedlerEmbeddedSpace(L,k):
	# L = Laplacian
	# k = dimension after dimension reduction

	# Perform Eigen Decomposition on the Laplacian matrix L where L = V * D * (VT) where VT is Transpose of V
	# V and D are the eigenvectors and eigenvalues 

	# Need the k+1  eigenvalues (non zero) and eigenvectors
	# ie the smallest eigenvalue is not included
	# Eigenvalues must be in increasing order
	
	evals, evecs = eigsh(L, (k+1), which='SM', maxiter=5000)
	# Note if you have scipy 0.11 consider using shift invert mode
	# evals_small, evecs_small = eigsh(X, 3, sigma=0, which='LM')
	
	sigma, eigenvects = eigsh(L, (k+1, which='SM', return_eigenvectors=True)
        fieldler_vector = sigma[1], X[:, 1]

	'''
	eval_k = []
	evecs_k = []
	for eval_index in range(1,len(evals)):
		eval_k.append(evals[eval_index])
		evecs_k.append(evecs[:,eval_index])

	eval_k = array(eval_k)
	evecs_k = array(evecs_k).T
	'''
	eval_k = fieldler_vector[0]
	evecs_k = fieldler_vector[1].T
	# Make S the k-dimensional embedded space S = (Dk^0.5) * VkT
	# where Dk and Vk are the k eigenvalues and corresponding
	# Should only real values be returned?
	# when evecs_k**0.5 is used I get nan?
	# Is this the correct equation?
	eval_k = diag(eval_k,0)**0.5
	S = dot(eval_k,evecs_k.T)
	return S	

def query(S,q):
	'''Takes S the k-dimensional embedded space and the query vector as a parameter'''
	q_norm = q/linalg.norm(q) # normalize query vector
	qpos = dot(S,q_norm)
	return qpos

def knnMatches(S,qpos,K):
	""" find the K nearest neighbours of in embedded space S """
	qpos = qpos.T
	diff = (S.T - qpos)**2
	diff_sum = array(add.reduce(diff, axis=1))	
	diff_sum = diff_sum**0.5
	idx = argsort(diff_sum) 
	return idx[:K]

'''
# Example document-term matrix
# Sentences from http://web.eecs.utk.edu/~berry/order/node4.html#SECTION00022000000000000000
# nine docs c1- c5 related to human-computer interaction and m1- m4 related to graph theory.
# Docs:
# C1 = Human machine interface for Lab ABC computer applications
# C2 = A survey of user opinion of computer system response time
# C3 = The EPS user interface management system
# C4 = System and human system engineering testing of EPS
# C5 = Relation of user-perceived response time to error measurement
# M1 = The generation of random, binary, unordered trees
# M2 = The intersection graph of paths in trees
# M3 = Graph minors IV: Widths of trees and quasi-ordering
# M4 = Graph minors: A survey
'''
termanddocvect = ["computer", "EPS", "human", "interface", "response", "system", "time", "user", "minors", "survey","trees", "graph", "C1", "C2", "C3", "C4", "C5", "M1", "M2", "M3", "M4"]

# Vector dimensions: computer, EPS, human, interface, response, system, time, user, minors, survey,trees, graph
docterm=array([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0], 
		   [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
		   [0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
		   [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
		   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
		   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
		   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], 
		   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0], 
		   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]])

# Create Laplacian block Matrix
L = createLaplacian(docterm)

k = 2
S = fiedlerEmbeddedSpace(L,k)

print S

# query for human
q = array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
qpos = query(S,q)

matches = knnMatches(S,qpos,9)

for i in matches:
	print termanddocvect[i]

