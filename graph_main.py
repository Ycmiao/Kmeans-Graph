#import numpy as np
import argparse
import logging
from sklearn.datasets import make_circles
import numpy as np

from graph_models import  MMSBM

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

	alpha = 0.3 * np.array([1,1,1]) #np.random.rand(3)
	nVertices = 100
	#bernouilli_matrix = np.diag(np.ones(len(alpha)))
	bernouilli_matrix = np.ones(len(alpha))*0.1+np.diag([1,1,1])*0.8
	sparsity_param = 0.
	print('\nalpha = ', alpha)
	print('\nbernouilli maxtrix = \n', bernouilli_matrix)
	print('\nsparsity parameter = \n', sparsity_param)

	my_graph_model = MMSBM(alpha, nVertices, bernouilli_matrix, sparsity_param)
	my_graph_model.generateGraph()
	my_graph_model.showGraph()

	my_graph_model.groundTruthClustering()

	my_graph_model.kmeansClustering()

	my_graph_model.getDegreesMatrix()

	my_graph_model.spectralClustering()

	my_graph_model.opticsClustring()



