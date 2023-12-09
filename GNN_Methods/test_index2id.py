import pickle

with open("./yelp_test_data.pkl", 'rb') as f:
    idx_test, y_test = pickle.load(f)

with open("./rid_mapping.pkl", 'rb') as f:
    rid_mapping = pickle.load(f)

index_to_id = {v: k for k, v in rid_mapping.items()}
test_ids = [index_to_id[idx] for idx in idx_test if idx in index_to_id]

with open('./yelp_test_ids.pkl', 'wb') as f:
    pickle.dump(test_ids, f)