# Anomaly-detection-LLM

## Data generation
To generate the data:
```bash
python helper.py
```

Data generated:
-Original dataset before filter:
```bash
‘UNPRUNED_DATA_prod-ID_usr-ID_rating_label_review.json’
```
-Dataset after filter all the products with more than 20 reviews:
```bash
‘PRUNED_DATA_prod-ID_usr-ID_rating_label_review.json’
```

Data format:
`{prod_ID: user_ID, user_label, review_label, rating score, review text}`

-review_label is product specific: Created by Yelp company. For label value 1 means non-spam review and -1 means spam review. There are 13.23% spam reviews in total in the unpruned data.

-user_label is user specific: Created by us. For label value 1 means normal users (authors with no spam reviews) and -1 means fraud users (authors with at least one spam review). There are 20.33% spam users in total in unpruned data.

-Rating score is given by each user for one prodcut.

