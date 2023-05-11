#import numpy as np
import argparse
import logging
from sklearn.datasets import make_circles

from graph_2dmodels import Graph2D

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--log', default='DEBUG',
					help = 'log level in {DEBUG, INFO, WARNING, ERROR, CRITICAL}')
	args = ap.parse_args()

	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	file_handler = logging.FileHandler('info.log', mode='w')
	file_handler.setLevel(getattr(logging, args.log))
	logger.addHandler(file_handler)
	logger.info("hello world")


	# ----------- cloud graph -----------
	X_cloud, labels = make_circles(n_samples=100, noise=0.1, factor=.2)

	Cloud_graph = Graph2D(X_cloud, labels)

	Cloud_graph.generateMutualKNNGraph(10, sym='loose')

	Cloud_graph.showGraph()

	Cloud_graph.kmeansClustering()

	Cloud_graph.spectralClustering()

	Cloud_graph.opticsClustring()
