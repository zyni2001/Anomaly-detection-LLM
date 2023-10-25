# Anomaly-detection-LLM

## Data generation
-Original dataset before filter:
```bash
‘UNPRUNED_DATA_prod-ID_usr-ID_rating_label_review.json’
```

-Dataset after filter all the products with more than 20 reviews:
```bash
‘PRUNED_DATA_prod-ID_usr-ID_rating_label_review.json’
```

```bash
python helper.py
```