from numpy import *
from scipy import linalg,array,dot,mat
from math import *
from pprint import pprint

def query(S,q):
	'''Takes S the k-dimensional embedded space and the query vector as a parameter'''
	q_norm = q/linalg.norm(q) # normalize query vector
	qpos = S.transpose() * q_norm
	return qpos

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

# Vector dimensions: computer, EPS, human, interface, response, system, time, user, minors, survey,trees, graph
matrix=array([[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0], 
		   [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
		   [0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
		   [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
		   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
		   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 
		   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], 
		   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0], 
		   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]])

nodocs, noterms = matrix.shape
		   
# Create word-word mapping matrix
# Initially constructed from the doc-word matrix
#WW = zeros((noterms,noterms))	
WW = dot(matrix.transpose(),matrix)
# Create doc-doc mapping matrix	
#DD = zeros((nodocs,nodocs))
DD = dot(matrix,matrix.transpose())

B = matrix * (-1)
BT = B.transpose()

# Create Block Matrix L
# L is a (nodocs + noterms) by (nodocs + noterms) matrix
#  ---            ---
# |  WW    BT  |
# |  B        DD  |  
#  ---            ---
L = bmat('WW,BT; B,DD')

# k = dimension after dimension reduction
k = 3

# Perform Eigen Decomposition on the Laplacian matrix L where L = V * D * (VT) where VT is Transpose of V
# V and D are the eigenvectors and eigenvalues 
evals, evecs = linalg.eig(L)

# Sort Eigenvalues	
evals_sorted = sort(evals)

# Remove largest eigenvalue and get next k largest eigenvalues
no_eigenvalues = evals_sorted.shape[0]

kevals = evals_sorted[(no_eigenvalues-1-k):(no_eigenvalues-1)]

# Confused about how to get kevecs (should they match evals?)
kevecs = evecs[:, (no_eigenvalues-1-k):(no_eigenvalues-1) ]
	
# Make S the k-dimensional embedded space S = (Dk^0.5) * VkT
# where Dk and Vk are the k eigenvalues and corresponding
S = power(kevecs, 0.5) * kevals.transpose()

# Query - human + interface
# Query is constructed with first terms = words (1 to nodocs) and then documents (nodocs+1 to nodocs+1+nodocs)
# Is this the correct way to create the query? Can query on docs and words?
q1 = array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

qpos= query(S,q1)

print S
print qpos.transpose()

#ToDo - Find k-nearest neighbours