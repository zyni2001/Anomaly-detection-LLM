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

## Deploy vllm

### Install vllm
```bash
cd vllm/
pip install -e .
```

### Login to HuggingFace
```bash
huggingface-cli login
```
There will be a token needed. Copy and paste `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` to the terminal.

There will appear "Add token as git credential? (Y/n)". Enter `Y`.

### Quick start
Start the API server:
```bash
cd ../
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf
```

Query the model in another shell:
```bash
python examples/openai_chatcompletion_client.py 
```
Results will be saved in `results.txt`.

## Prompt for OD

```bash
python prompt.py
```