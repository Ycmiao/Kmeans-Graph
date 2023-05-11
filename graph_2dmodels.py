import numpy as np
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS,cluster_optics_dbscan
from itertools import permutations
import networkx as nx

class Graph2D:
	def __init__(self, cloudPoints, trueClusters):
		self.cloudPoints = cloudPoints
		self.adjacencyMat = None
		self.trueClusters = trueClusters
		self.K = len(np.unique(self.trueClusters))
		self.nVertices = len(self.cloudPoints)
		self.n_ngb=0
		# rearrange vectors based on labels
		self.cloudPoints=np.vstack( (self.cloudPoints[self.trueClusters==0,:], self.cloudPoints[self.trueClusters==1,:]))
		self.trueClusters=np.hstack( (self.trueClusters[self.trueClusters==0], self.trueClusters[self.trueClusters==1]))

	def generateMutualKNNGraph(self, n_ngb=5, sym='tight'):
		self.n_ngb = n_ngb
		self.adjacencyMat = kneighbors_graph(self.cloudPoints, n_neighbors=n_ngb, include_self=True).toarray()
		# make adjacency matrix symmetrical
		if sym=='loose':
			self.adjacencyMat = np.maximum(self.adjacencyMat,self.adjacencyMat.transpose())
		else:
			self.adjacencyMat = np.multiply(self.adjacencyMat,self.adjacencyMat.transpose())

	def showGraph(self):
		plt.figure(figsize=(12,6))

		# drawing adjacency matrix
		plt.subplot(131)
		print(self.nVertices)
		matPlot = np.zeros((self.nVertices, self.nVertices, 3))
		matPlot[self.adjacencyMat==0] = np.array([255,255,255])
		matPlot[self.adjacencyMat>0] = np.array([255, 0, 255])#([0, 153, 255])
		plt.imshow(matPlot)
		plt.title("adjacency matrix")
		# plotting cloud data points
		plt.subplot(132)
		plt.scatter(self.cloudPoints[:,0],self.cloudPoints[:,1], c=['red' if l==0 else 'blue' for l in self.trueClusters ])
		plt.axis('off')
		plt.title('point cloud data')
		# plot edges
		plt.subplot(133)
		plt.scatter(self.cloudPoints[:,0],self.cloudPoints[:,1], c=['red' if l==0 else 'blue' for l in self.trueClusters ])
		plt.axis('off')
		plt.title('{}-NN graph on PCD'.format(self.n_ngb))
		for i in range(self.nVertices-1):
			for j in range(i+1,self.nVertices):
				if self.adjacencyMat[i,j]==1:
					plt.plot(self.cloudPoints[[i,j],0], self.cloudPoints[[i,j],1], 'k-', linewidth=0.2)
		plt.show()


	def MatchingGraph(self,resClusters,methods):
		# matching clusters labels with ground truth clusters labels
		bestSimil=0
		for perm in list(permutations(list(range(self.K)))):
			permKClusters=np.zeros(len(self.trueClusters),dtype=np.int8)
			permKClusters[resClusters.labels_==0]=perm[0]
			permKClusters[resClusters.labels_==1]=perm[1]
			simil = np.sum(permKClusters==self.trueClusters)
			if simil>bestSimil:
				bestSimil=simil
				matchedKClusters=permKClusters
		resMatchedClusters = np.array(matchedKClusters)
		print(methods+' on adjacency matrix > label recovery error: {:.1f}%\n'.format(100*(self.nVertices-bestSimil)/self.nVertices))

		return resMatchedClusters

	def kmeansClustering(self):
		plt.figure(figsize=(13,6))

		# infer kmeans clusters from PCD
		kmeans = KMeans(n_clusters=self.K, random_state=0).fit(self.cloudPoints)
		kmeansClusters = self.MatchingGraph(kmeans, 'Kmeans')

		plt.subplot(121)
		plt.scatter(self.cloudPoints[:,0],self.cloudPoints[:,1], c=['red' if l==0 else 'blue' for l in kmeansClusters])
		plt.axis('off')
		plt.title('Kmeans clustering on PCD')

		# infer kmeans clusters from points in adjacency representation
		kmeans = KMeans(n_clusters=self.K, random_state=0).fit(self.adjacencyMat)


		plt.subplot(122)
		plt.scatter(self.cloudPoints[:,0],self.cloudPoints[:,1], c=['red' if l==0 else 'blue' for l in kmeansClusters ])
		plt.axis('off')
		plt.title('Kmeans clustering on adjacency matrix')

		plt.show()
	def opticsClustring(self, colorClusters=['pink', 'cyan', 'orange', 'green', 'purple', 'brown']):
		# infer optics clusters
		optics = OPTICS(min_samples=self.K, xi=0.5, min_cluster_size=0.5,cluster_method='xi').fit(self.cloudPoints)

		# matching optics clusters labels with ground truth clusters labels
		opticsClusters = self.MatchingGraph(optics, 'Optics_ST')
		plt.subplot(121)
		plt.scatter(self.cloudPoints[:, 0], self.cloudPoints[:, 1],
					c=['red' if l == 0 else 'blue' for l in opticsClusters])
		plt.axis('off')
		plt.title('Optics_ST clustering')

		labels_200 = cluster_optics_dbscan(
			reachability=optics.reachability_,
			core_distances=optics.core_distances_,
			ordering=optics.ordering_,
			eps=2,
		)
		# optics = OPTICS(min_samples=self.K, eps=0.5, min_cluster_size=0.5,cluster_method='dbscan').fit(self.cloudPoints)

		# matching optics clusters labels with ground truth clusters labels
		opticsClusters = self.MatchingGraph(optics, 'Optics_ST')
		plt.subplot(122)
		plt.scatter(self.cloudPoints[:, 0], self.cloudPoints[:, 1],
					c=['red' if l == 0 else 'blue' for l in opticsClusters])
		plt.axis('off')
		plt.title('Optics_DBSCAN clustering')

		plt.show()

	def getDegreesMatrix(self):
		diag = []
		for i in range(self.nVertices):
			diag.append(np.sum(self.adjacencyMat[i,:]))
		return np.diag(diag)

	def spectralClustering(self):

		Laplacian = self.getDegreesMatrix() - self.adjacencyMat
		Lsym = np.diag(np.diag(self.getDegreesMatrix())**-0.5) @ Laplacian @ np.diag(np.diag(self.getDegreesMatrix())**-0.5)

		ev, U = np.linalg.eig(Lsym)

		plt.figure(figsize=(13,6))
		plt.subplot(131)
		plt.scatter(list(range(self.nVertices)), ev, marker = 'd',c='red')
		plt.title('eigenvalues')

		plt.subplot(132)
		plt.plot(U[:,0],'-o', label='eigenvector 1')
		plt.plot(U[:,1],'-o', label='eigenvector 2')
		plt.legend(loc='upper right')

		T_norm = np.tile(1./np.sqrt(np.sum(U[:,:self.K]**2, axis=1)),(self.K,1)).transpose() # row normalization
		X_spec_norm = np.multiply(U[:,:self.K], T_norm)

		# repeat same steps as for KMeans
		# infer kmeans clusters
		kmeans = KMeans(n_clusters=self.K, random_state=0).fit(X_spec_norm)
		#self.kmeansClusters=np.array(self.K-1-kmeans.labels_,dtype=np.int8)

		# matching kmeans clusters labels with ground truth clusters labels
		kmeansClusters = self.MatchingGraph(kmeans, 'Spectral')
		plt.subplot(133)
		plt.scatter(self.cloudPoints[:,0],self.cloudPoints[:,1], c=['red' if l==0 else 'blue' for l in kmeansClusters ])
		plt.axis('off')
		plt.title('Spectral clustering')

		plt.show()