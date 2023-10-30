# Anomaly-detection-LLM

## Data

### Data generation
To generate the data:
```bash
cd Data/
python helper.py
```

Data generated:
- Original dataset before filter:
```bash
‘UNPRUNED_DATA_prod-ID_usr-ID_rating_label_review.json’
```
- Dataset after filter all the products with more than 20 reviews:
```bash
‘PRUNED_DATA_prod-ID_usr-ID_rating_label_review.json’
```

### Data format:
`{prod_ID: user_ID, user_label, review_label, rating score, review text}`

- `review_label` - Created by Yelp company. For label value 1 means non-spam review and -1 means spam review. There are 13.23% spam reviews in total in the unpruned data.
- `user_label` - Created by us. For label value 1 means normal users (authors with no spam reviews) and -1 means fraud users (authors with at least one spam review). There are 20.33% spam users in total in unpruned data.

- `rating score` - given by `user_ID` to `prod_ID`.

- `review text` - the review content.

