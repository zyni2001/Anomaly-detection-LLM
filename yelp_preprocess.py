import numpy as np
import scipy.sparse as sp
import gzip
import copy as cp
from datetime import datetime


def read_graph_data(metadata_filename, graph_data_structure='up'):
	""" Read the user-review-product graph from file. Can output the graph in different formats

		Args:
			metadata_filename: a gzipped file containing the graph.
			graph_data_structure: defines the output graph format
				'up' (default) ---> user-product and product-user graphs
				'urp' ---> user-review and review-product graphs
		Return:
			graph1: user-product / user-review
			graph2: product-user / review-product
	"""

	user_data = {}

	prod_data = {}

	# use the rt mode to read ascii strings instead of binary
	with gzip.open(metadata_filename, 'rt') as f:
		# file format: each line is a tuple (user id, product id, rating, label, date)
		for line in f:
			items = line.strip().split()
			u_id = items[0]
			p_id = items[1]
			rating = float(items[2])
			label = int(items[3])
			date = items[4]

			if u_id not in user_data:
				user_data[u_id] = []
			user_data[u_id].append((p_id, rating, label, date))

			if p_id not in prod_data:
				prod_data[p_id] = []
			prod_data[p_id].append((u_id, rating, label, date))

	# read text feature files, including: wordcount, ratio of SW/OW, etc.
	# constructed by the python files provided by the authors.
	print('read reviews from %s' % metadata_filename)
	print('number of users = %d' % len(user_data))
	print('number of products = %d' % len(prod_data))

	if graph_data_structure == 'up':
		return user_data, prod_data

	if graph_data_structure == 'urp':
		user_review_graph = {}
		for k, v in user_data.items():
			user_review_graph[k] = []
			for t in v:
				user_review_graph[k].append((k, t[0]))  # (u_id, p_id) representing a review
		review_product_graph = {}
		for k, v in prod_data.items():
			for t in v:
				# (u_id, p_id) = (t[0], k) is the key of a review
				review_product_graph[(t[0], k)] = k
		return user_review_graph, review_product_graph


def remove_reviews(upg, pug):

	user_prod_graph = cp.deepcopy(upg)
	prod_user_graph = cp.deepcopy(pug)

	removed_prod = []
	removed_user = []
	for prod, reviews in prod_user_graph.items():
		if len(reviews) > 800:
			removed_prod.append(prod)

	for user, reviews in user_prod_graph.items():
		for review in reviews:
			if review[0] in removed_prod:
				user_prod_graph[user].remove(review)

	for user, reviews in user_prod_graph.items():
		if len(reviews) == 0:
			removed_user.append(user)

	print('number of removed users = %d' % len(removed_user))
	print('number of removed products = %d' % len(removed_prod))
	
	return removed_user, removed_prod


def load_new_graph(metadata_filename, removed_user, removed_prod):

	user_data = {}

	prod_data = {}

	# use the rt mode to read ascii strings instead of binary
	with gzip.open(metadata_filename, 'rt') as f:
		# file format: each line is a tuple (user id, product id, rating, label, date)
		for line in f:
			items = line.strip().split()
			u_id = items[0]
			p_id = items[1]
			rating = float(items[2])
			label = int(items[3])
			date = items[4]

			if u_id not in removed_user and p_id not in removed_prod:
				if u_id not in user_data:
					user_data[u_id] = []
				user_data[u_id].append((p_id, rating, label, date))

				if p_id not in prod_data:
					prod_data[p_id] = []
				prod_data[p_id].append((u_id, rating, label, date))

	# read text feature files, including: wordcount, ratio of SW/OW, etc.
	# constructed by the python files provided by the authors.
	print('read reviews from %s' % metadata_filename)
	print('number of users = %d' % len(user_data))
	print('number of products = %d' % len(prod_data))

	return user_data, prod_data


def create_ground_truth(user_data):
	"""Given user data, return a dictionary of labels of users and reviews
	Args:
		user_data: key = user_id, value = list of review tuples.

	Return:
		user_ground_truth: key = user id (not prefixed), value = 0 (non-spam) /1 (spam)
		review_ground_truth: review id (not prefixed), value = 0 (non-spam) /1 (spam)
	"""
	user_ground_truth = {}
	review_ground_truth = {}

	for user_id, reviews in user_data.items():

		user_ground_truth[user_id] = 0

		for r in reviews:
			prod_id = r[0]
			label = r[2]

			if label == -1:
				review_ground_truth[(user_id, prod_id)] = 1
				user_ground_truth[user_id] = 1
			else:
				review_ground_truth[(user_id, prod_id)] = 0

	return user_ground_truth, review_ground_truth


def time_judge(time1, time2):

	date1 = datetime.strptime(time1, '%Y-%m-%d')
	date2 = datetime.strptime(time2, '%Y-%m-%d')

	if date1.year == date2.year and date1.month == date2.month:
		return True
	else:
		return False


def meta_to_homo(meta_data_name):
	"""
	generating homogeneous adjacency matrix from metadata
	:return:
	"""

	upg, pug = read_graph_data(meta_data_name)

	removed_user, removed_prod = remove_reviews(upg, pug)

	user_prod_graph, prod_user_graph = load_new_graph(meta_data_name, removed_user, removed_prod)

	user_ground_truth, review_ground_truth = create_ground_truth(user_prod_graph)

	# map review id to adj matrix id
	rid_mapping = {}
	r_index = 0
	for review in review_ground_truth.keys():
		rid_mapping[review] = r_index
		r_index += 1

	review_adj = sp.lil_matrix((len(review_ground_truth), len(review_ground_truth)))

	# review-review graph with stacked multiple relations
	# 1) r-product-r
	# for p, reviews in prod_user_graph.items():
	# 	for r0 in reviews:
	# 		for r1 in reviews:
	# 			if r0[0] != r1[0]:  # do not add self loop at this step
	# 				review_adj[rid_mapping[(r0[0], p)], rid_mapping[(r1[0], p)]] = 1
	# 2) r-user-r
	# for u, reviews in user_prod_graph.items():
	# 	for r0 in reviews:
	# 		for r1 in reviews:
	# 			if r0[0] != r1[0]:
	# 				review_adj[rid_mapping[(u, r0[0])], rid_mapping[(u, r1[0])]] = 1
	# 3) r-time-r
	for p, reviews in prod_user_graph.items():
		for r0 in reviews:
			for r1 in reviews:
				if r0[0] != r1[0] and time_judge(r0[3], r1[3]) == True:
					review_adj[rid_mapping[(r0[0], p)], rid_mapping[(r1[0], p)]] = 1
	# # 4) r-star-r
	# for p, reviews in prod_user_graph.items():
	# 	for r0 in reviews:
	# 		for r1 in reviews:
	# 			if r0[0] != r1[0] and r0[1] == r1[1]:
	# 				review_adj[rid_mapping[(r0[0], p)], rid_mapping[(r1[0], p)]] = 1

	# # 5) r-star/time-r
	# for p, reviews in prod_user_graph.items():
	# 	for r0 in reviews:
	# 		for r1 in reviews:
	# 			if r0[0] != r1[0] and r0[1] == r1[1] and time_judge(r0[3], r1[3]) == True:
	# 				review_adj[rid_mapping[(r0[0], p)], rid_mapping[(r1[0], p)]] = 1

	review_adj = review_adj.tocsc()

	return review_adj


if __name__ == "__main__":

	prefix = './'
	meta_data_name = prefix + 'metadata.gz'
	review_adj_name = prefix + 'review_rur_adj.npz'

	# generate homo adjacency matrix
	review_adj = meta_to_homo(meta_data_name)
	# sp.save_npz(review_adj_name, review_adj)
