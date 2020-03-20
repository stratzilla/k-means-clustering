#!/usr/bin/env python3

from sys import argv, exit
from math import sqrt, inf
from random import choice
from os.path import isfile, isdir
from os import system

try:
	import pandas as pd # some containerization for input data
	import matplotlib.pyplot as plt # plotting
except ImportError:
	print("\nModules could not be loaded.")
	print("Ensure module dependencies are installed before execution.\n")
	exit(1)

def execution_instructions():
	"""Reminds user how to execute on error."""
	print("\nExecute the script as the below:")
	print(" $ ./k-means.py <File> <Type> <K> <Epochs>\n")
	print("Where the following are the arguments:")
	print(" <File> -- the data to perform k-means clustering upon")
	print(" <Type> -- the type of metric space to use:")
	print("        -- 1 - Euclidean Metric")
	print("        -- 2 - Manhattan Metric")
	print("        -- 3 - Chebyshev Metric")
	print(" <K> -- k-parameter, or how many clusters [2, 25]")
	print(" <Epochs> -- how many epochs for k-means to perform [1, 100]\n")
	print("Each epoch frame will be saved as an image in `./plots/`.")
	print("This directory will be made if it does not exist.\n")
	exit(1)

try:
	if len(argv) != 5:
		raise ValueError
	FILENAME = argv[1]
	if not isfile(FILENAME):
		raise FileNotFoundError
	TYPE = int(argv[2])
	K_PARAM = int(argv[3])
	MAX_EPOCH = int(argv[4])
	OUTPUT = None
	if not 3 >= TYPE >= 1 or not 25 >= K_PARAM >= 2 or MAX_EPOCH < 1:
		raise ValueError
	if not isdir('./plots/'):
		system('mkdir plots')
except (ValueError, FileNotFoundError):
	execution_instructions()

class Point:
	"""Point class.

	Containerizes an x,y-coordinate along with cluster data.

	Attributes:
		x_c : the x-coordinate of the Point.
		y_c : the y-coordinate of the Point.
		cluster : which cluster the Point belongs to.
	"""

	def __init__(self, x, y):
		"""Constructs Point based on x,y-coordinate."""
		self.x_c, self.y_c, self.cluster = x, y, None

	def set_coord(self, x, y):
		"""x,y-coordinate mutator method."""
		self.x_c, self.y_c = x, y

	def get_coord(self):
		"""x,y-coordinate accessor method."""
		return self.x_c, self.y_c

	def set_cluster(self, c):
		"""cluster mutator method."""
		self.cluster = c

	def get_cluster(self):
		"""cluster accessor method."""
		return self.cluster

	def __eq__(self, other):
		"""Equality overload method."""
		return self.x_c == other.x_c and self.y_c == other.y_c

	def __neq__(self, other):
		"""Inequality overload method."""
		return not self.__eq__(other)

def k_means(epochs=MAX_EPOCH):
	"""Performs the k-means clustering on the data.

	The k-means algorithm generates centroids within a data set which are
	maximally distant from each other. Then, finds the nearest points to each
	centroid and considers them members of that cluster. The centroids are then
	moved where the new position is the mean position of each member point. The
	process of finding nearest membership points and moving centroids is
	repeated over many epochs/iterations, eventually the centroids will converge
	to the center of clusters within the data set.

	Parameters:
		epochs : how many iterations the algorithm should perform.
	"""
	points = load_data()
	clusters = initialize_centroids(points)
	find_clusters(points, clusters)
	plot_data(points, clusters, 0)
	for i in range(epochs):
		find_clusters(points, clusters)
		move_centroids(points, clusters)
		plot_data(points, clusters, i+1)
	print(f"\nThe Dunn Index is: {dunn_index(points, clusters)}.\n")

def dunn_index(points, clusters):
	"""Calculates the Dunn Index of the clustering.

	The Dunn Index is a function of the spread of clusters and the spread of
	points within a cluster. DI = min cluster distance / max cluster diameter.
	A smaller Dunn Index indicates better clustering.

	Parameters:
		points : a list of Points.
		clusters : a list of cluster centroids.

	Returns:
		The Dunn Index of the clustering.
	"""
	min_cluster_dist = inf
	# compare distances between cluster centroids, choose smallest
	for c_i in clusters:
		for c_j in clusters:
			if c_i is c_j:
				continue # distance between cluster and itself is 0
			min_cluster_dist = min(min_cluster_dist, distance(c_i, c_j))
	max_cluster_diam = 0
	# compare diameters of each cluster, choose largest
	for c in clusters:
		for p_i in points:
			if p_i.get_cluster() is not c:
				continue # if not in cluster
			for p_j in points:
				if p_j.get_cluster() is not c:
					continue # if not in cluster
				if p_i is p_j:
					continue # distance between point and itself is 0
				max_cluster_diam = max(max_cluster_diam, distance(p_i, p_j))
	return round(min_cluster_dist/max_cluster_diam, 4)

def distance(p, q, t=TYPE):
	"""Distance calculator.

	Determine the distance between points based on CLI argument. Included metric
	spaces are Euclidean, Manhattan (Rectilinear), and Chebyshev.

	Parameters:
		p : Point p.
		q : Point q.
		t : the type of metric to use.

	Returns:
		The distance between Points p and q.
	"""
	(p_x, p_y), (q_x, q_y) = p.get_coord(), q.get_coord()
	# Euclidean Distance measure, d(p, q) = sqrt((q_1 - p_1)^2 + (q_2 - p_2)^2)
	if t == 1:
		return sqrt((q_x - p_x)**2 + (q_y - p_y)**2)
	# Manhattan Distance measure, d(p, q) = |q_1 - p_1| + |q_2 - p_2|
	if t == 2:
		return abs(q_x - p_x) + abs(q_y - p_y)
	# Chebyshev Distance measure, d(p, q) = max(|q_1 - p_1|, |q_2 - p_2|)
	return max(abs(q_x - p_x), abs(q_y - p_y))

def find_clusters(points, clusters):
	"""Pairs each Point with its closest neighbor cluster.

	Finds cluster based on distance between Point and cluster centroid.

	Parameters:
		points : a list of Points.
		clusters : a list of cluster centroids.
	"""
	for p in points:
		best_distance, best_cluster = inf, None
		for c in clusters:
			d = distance(p, c)
			if d < best_distance:
				best_distance, best_cluster = d, c
		p.set_cluster(best_cluster)

def initialize_centroids(points, k=K_PARAM):
	"""Use k-means++ as initialization strategy for centroids.

	Initializes centroid initial locations based on the distances between points
	and previously found centroids. Ensures maximal distance is between
	centroids which allows for a better k-means algorithm performance.

	Parameters:
		points : a list of Points.
		k : how many centroids to place.

	Returns:
		A list of centroids.
	"""
	clusters = []
	clusters.append(choice(points)) # first centroid is random point
	for _ in range(k - 1): # for other centroids
		distances = []
		for p in points:
			d = inf
			for c in clusters: # find the minimal distance between p and c
				d = min(d, distance(p, c))
			distances.append(d)
		# find maximum distance index from minimal distances
		clusters.append(points[distances.index(max(distances))])
	return clusters

def move_centroids(points, clusters):
	"""Repositions cluster centroids to the mean Point position.

	Parameters:
		points : a list of Points.
		clusters : a list of cluster centroids.
	"""
	for c in clusters:
		mean_x, mean_y, count = 0, 0, 0
		for p in points:
			if p.get_cluster() is c: # if Point belongs to cluster
				# aggregate the mean position of cluster Points
				mean_x += p.get_coord()[0]
				mean_y += p.get_coord()[1]
				count += 1
		# move centroid to mean position of Points in cluster
		c.set_coord(mean_x/count, mean_y/count)

def load_data(file=FILENAME):
	"""Loads data from file and inserts into list of Points.

	Parameters:
		file : filename of local file to parse Points from.

	Returns:
		A list of Points.
	"""
	df = pd.read_csv(file, names=['x', 'y'], delimiter=r'\s+')
	arr_points = []
	# convert from pd df to Points, append to list
	for _, d in df.iterrows():
		arr_points.append(Point(d['x'], d['y']))
	return arr_points

def plot_data(points, clusters, epoch):
	"""Plots data and cluster centroids.

	Parameters:
		points : a list of Points.
		clusters : a list of cluster centroids.
		epoch : iteration of the k-means algorithm.
	"""
	plt.xticks([])
	plt.yticks([])
	plt.margins(0.05, 0.05)
	# colors chosen to increase separability/improve distinction
	colors = ['red', 'lime', 'blue', 'yellow', 'orange', 'deeppink', \
		'olivedrab', 'aqua', 'thistle', 'mediumvioletred', 'plum', \
		'burlywood', 'maroon', 'mediumspringgreen', 'dodgerblue', \
		'rebeccapurple', 'lightcoral', 'darkslategrey', 'firebrick', 'bisque', \
		'darkseagreen', 'fuchsia', 'turquoise', 'steelblue', 'chocolate']
	for c, i in zip(clusters, range(len(clusters))):
		point_x, point_y = [], []
		for p in points:
			if p.get_cluster() is not c:
				continue
			x, y = p.get_coord()
			point_x.append(x)
			point_y.append(y)
		plt.scatter(point_x, point_y, c=colors[i], s=2, marker='o', lw='1')
	for c, i in zip(clusters, range(len(clusters))):
		x, y = c.get_coord()
		plt.scatter(x, y, c=colors[i], s=100, marker='X', lw=1, ec='k')
	save_destination = f"./plots/{epoch:03d}.png"
	plt.savefig(save_destination, bbox_inches='tight', pad_inches=0.1)
	plt.clf()

if __name__ == '__main__':
	k_means()
	exit(0)
