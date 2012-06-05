from numpy import *
from scipy import linalg,array,dot,mat
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
	WW = dot(DoctermMatrix.transpose(),DoctermMatrix)
	# Create doc-doc mapping matrix	
	DD = dot(DoctermMatrix,DoctermMatrix.transpose())
	B = DoctermMatrix * (-1)
	BT = B.transpose()

	# Create Block Matrix L
	# L is a (nodocs + noterms) by (nodocs + noterms) matrix
	#  ---      ---
	# WW    BT  
	# B        DD
	#  ---      ---
	L = bmat('WW,BT; B,DD')
	return L


def fiedlerEmbeddedSpace(L,k):
	# L = Laplacian
	# k = dimension after dimension reduction

	# Perform Eigen Decomposition on the Laplacian matrix L where L = V * D * (VT) where VT is Transpose of V
	# V and D are the eigenvectors and eigenvalues 
	evals, evecs = linalg.eig(L)

	#Store eigenvalues in a dictionary so they can be sorted but the index can still be obtained
	# Need the k smallest eigenvalues (non zero) and eigenvectors
	evaldict = {}
	count = 0
	for eval in evals:
		evaldict[count] = eval
		count = count + 1
	ordered_eval_list = sorted(evaldict, key=evaldict.get, reverse=True)

	eval_k_index = []
	assigned_eval = 0;
	for i in range(len(ordered_eval_list)):
		curr_eval = evals[ordered_eval_list[i]]
		if (curr_eval  != 0):
			assigned_eval = assigned_eval + 1
			eval_k_index.append(ordered_eval_list[i])
		if (assigned_eval==k):
			break

	eval_k = []
	evecs_k = []
	for eval in range(len(eval_k_index)):
		col_index = eval_k_index[eval]
		eval_k.append(evals[col_index])
		evecs_k.append(evecs[col_index,:])

	eval_k = array(eval_k)
	evecs_k = array(evecs_k).T

	# Make S the k-dimensional embedded space S = (Dk^0.5) * VkT
	# where Dk and Vk are the k eigenvalues and corresponding
	# Should only real values be returned?
	# when evecs_k**0.5 is used I get nan?
	# Is this the correct equation?
	S = ((evecs_k**0.5) * eval_k.transpose()).real
	return S	
	

def query(S,q):
	'''Takes S the k-dimensional embedded space and the query vector as a parameter'''
	q_norm = q/linalg.norm(q) # normalize query vector
	qpos = dot(S.transpose(),q_norm)
	return qpos

def knnMatches(S,qpos,K):
	""" find the K nearest neighbours of in embedded space S """
	qpos = qpos.T
	S = S.T
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

q = array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
qpos = query(S,q)

matches = knnMatches(S,qpos,9)

for i in matches:
	print termanddocvect[i]

print S.T