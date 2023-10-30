import copy as cp
import gzip
import json
# from torch_geometric.nn import DiffPool

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
	print('=' * 50)
	print('Original data')
	print('read reviews from %s' % metadata_filename)
	print('number of users = %d' % len(user_data))
	print('number of products = %d' % len(prod_data))
	print('=' * 50)

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


def remove_reviews(upg, pug, threshold=20, prune=True):

	user_prod_graph = cp.deepcopy(upg)
	prod_user_graph = cp.deepcopy(pug)

	removed_prod = []
	removed_user = []
	for prod, reviews in prod_user_graph.items():
		if len(reviews) > threshold and prune:
			removed_prod.append(prod)

	for user, reviews in user_prod_graph.items():
		for review in reviews:
			if review[0] in removed_prod:
				user_prod_graph[user].remove(review)

	for user, reviews in user_prod_graph.items():
		if len(reviews) == 0:
			removed_user.append(user)

	return removed_user, removed_prod


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

		user_ground_truth[user_id] = 1

		for r in reviews:
			prod_id = r[0]
			label = r[2]

			if label == -1:
				review_ground_truth[(user_id, prod_id)] = 1
				user_ground_truth[user_id] = -1
			else:
				review_ground_truth[(user_id, prod_id)] = 0

	return user_ground_truth, review_ground_truth


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
	print('=' * 50)
	print('Pruned data')
	print('read reviews from %s' % metadata_filename)
	print('number of users = %d' % len(user_data))
	print('number of products = %d' % len(prod_data))
	print('=' * 50)

	return user_data, prod_data


def load_text_data(hot_text_name, res_text_name, removed_user, removed_prod, review_ground_truth):

	# load raw text
	text_list = []
	with open(hot_text_name, 'rt') as f_in:
		for line in f_in.readlines():
			text_list.append(line)
	with open(res_text_name, 'rt') as f_in:
		for line in f_in.readlines():
			text_list.append(line)

	# match review text and review id
	review_text_mapping = {}
	line_index = 0
	with open(graph_meta_name, 'rt') as f_in:
		for line in f_in.readlines():
			line = line.split()
			u_id = line[0]
			p_id = line[1]

			if u_id not in removed_user and p_id not in removed_prod:
				review_text_mapping[(u_id, p_id)] = text_list[line_index]

			line_index += 1

	review_text_list = []
	for review in review_ground_truth.keys():
		review_text_list.append(review_text_mapping[review])

	return review_text_list, review_text_mapping

def create_prod_text_mapping(new_prod_data, review_text_mapping, user_ground_truth=None):
    prod_text_mapping = {}
    
    for prod_id, reviews in new_prod_data.items():
        prod_text_mapping[prod_id] = []
        
        for r in reviews:
            user_id = r[0]
            rating = r[1]
            label_review = r[2]
            label_user = user_ground_truth.get(user_id, None)

            # Fetch the review using the user_id and prod_id
            review_text = review_text_mapping.get((user_id, prod_id), None)

            # If the review text exists, append it to the prod_text_mapping
            if review_text:
                prod_text_mapping[prod_id].append((user_id, label_user, label_review, rating, review_text))
    
    return prod_text_mapping


if __name__ == '__main__':
	
	metadata_filename = 'metadata.gz'
	graph_meta_name = 'metadata.txt'
	hot_text_name = 'raw_text.txt'
	res_text_name = 'output_review_yelpResData_NRYRcleaned.txt'
	user_data, prod_data = read_graph_data(metadata_filename) # read the user-review-product dict from file

	""" create our dataset without pruning"""
	# create ground truth
	removed_user, removed_prod = remove_reviews(user_data, prod_data, prune=False) # remove no reviews
	new_user_data, new_prod_data = load_new_graph(metadata_filename, removed_user, removed_prod) # load the user-review-product dict from file and do pruning too
	user_ground_truth, review_ground_truth = create_ground_truth(new_user_data) # 0 (non-spam) /1 (spam)
	# calculate the percentage of spam users
	spam_users = 0
	for user_id, label in user_ground_truth.items():
		if label == -1:
			spam_users += 1
	percentage_spam_user = spam_users / len(user_ground_truth)
	# load review text and create prod_text_mapping
	_, review_text_mapping = load_text_data(hot_text_name, res_text_name, removed_user, removed_prod, review_ground_truth)
	prod_text_mapping = create_prod_text_mapping(new_prod_data, review_text_mapping, user_ground_truth)
	# save review_text_list as json file
	with open('UNPRUNED_DATA_prod-ID_usr-ID_rating_label_review.json', 'w') as fp:
		json.dump(prod_text_mapping, fp)


	""" create our dataset with pruning"""
	removed_user, removed_prod = remove_reviews(user_data, prod_data, threshold=20, prune=True) # remove products with more than 20 reviews and its corresponding users
	new_user_data, new_prod_data = load_new_graph(metadata_filename, removed_user, removed_prod) # load the user-review-product dict from file and do pruning too
	user_ground_truth, review_ground_truth = create_ground_truth(new_user_data)
	_, review_text_mapping = load_text_data(hot_text_name, res_text_name, removed_user, removed_prod, review_ground_truth)
	prod_text_mapping = create_prod_text_mapping(new_prod_data, review_text_mapping, user_ground_truth)
	with open('PRUNED_DATA_prod-ID_usr-ID_rating_label_review.json', 'w') as fp:
		json.dump(prod_text_mapping, fp)


	""" test the generated data """
	with open('UNPRUNED_DATA_prod-ID_usr-ID_rating_label_review.json', 'r') as fp:
		prod_text_mapping = json.load(fp)
	# calculate the total number of reviews
	total_num_reviews = 0
	for prod_id, reviews in prod_text_mapping.items():
		total_num_reviews += len(reviews)
	print('total number of reviews: %d' % total_num_reviews)
	# calculate the percentage of spam reviews
	spam_reviews = 0
	for prod_id, reviews in prod_text_mapping.items():
		for r in reviews:
			if r[2] == -1:
				spam_reviews += 1
	print('percentage of spam reviews in orignal dataset: %f' % (spam_reviews / total_num_reviews))
	# calculate the percentage of spam users
	print('percentage of spam users in orignal dataset: %f' % percentage_spam_user)


