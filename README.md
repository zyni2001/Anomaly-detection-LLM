# LLMs for Graph Anomaly Detection

> Author: Lumingyuan Tang, Chen Peng, Xingjian Dong, Zhiyu Ni

We introduced the Large Language Model method for graph anomaly detection, and we initially used the Chain of Thought (CoT) to perform pure inference on the test set and obtained fair experimental results. This code repository contains the main prompt scripts for LLMs and all the evaluation codes.

## Installation

To start, firstly clone the repo:

```bash
git clone https://github.com/zyni2001/Anomaly-detection-LLM
cd Anomaly-detection-LLM
```

Then install the dependencies into your environment:

```bash
pip install -r requirements.txt
```

Note that `torch==1.12.1+cu113` and `dgl-cu113==0.8.1` is not the lastest version and may need to be installed separatedly.

## Code Structure

The overall code structure shows below:

```shell
.
├── Data
├── GNN_Methods
│   ├── CARE-GNN
│   ├── PC-GNN
│   ├── antifraud
│   ├── rid_mapping.pkl
│   ├── test_index2id.py
│   ├── yelp_test_data.pkl
│   ├── yelp_test_ids.pkl
│   └── yelp_train_data.pkl
├── README.md
├── eval_gpt.py
├── eval_llama.py
├── llm_output
├── output
├── plot.py
├── prompt.py
├── prompt_llama.py
├── requirements.txt
└── vllm
```

- `Data/`: This folder contains the orignal text-based dataset from YelpCHI and Amazon as well as the preprocessing code. The LLMs' inference is mainly based on the data here.
- `GNN_Methods/`: Contains our baseline GNN-based methods code and pickle data for test data consistency.
- `eval_gpt.py` & `eval_llama.py`: Main scripts for evaluating LLM-generated outputs.
- `llm_out/`: The output .json files of LLMs. Each contains all the results and reasons about the anomalies LLMs finds.
- `output/`: Including the evaluation result figures and logs.
- `prompt.py` & `prompt_llama.py`: Prompt script for LLMs.
- `vllm/`: vllm package source code used for deploying self-used llama.

## Usage

### Dataset

Unlike the dataset based on Feature Embedding used by most GNN-based methods, we need to use the raw text data from the YelpCHI and Amazon datasets, therefore, here is provided a script for processing the raw data.

#### Data generation

To generate the data:

```bash
cd Data/
python helper.py
python yelp_preprocess.py
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

- Mapping dictionary (will be located in `./GNN_Methods/`):

```bash
‘rid_mapping.pkl’
```

#### Data format

`{prod_ID: user_ID, user_label, review_label, rating score, review text}`

- `review_label` - Created by Yelp company. For label value 1 means non-spam review and -1 means spam review. There are 13.23% spam reviews in total in the unpruned data.
- `user_label` - Created by us. For label value 1 means normal users (authors with no spam reviews) and -1 means fraud users (authors with at least one spam review). There are 20.33% spam users in total in unpruned data.
- `rating score` - given by `user_ID` to `prod_ID`.
- `review text` - the review content.

### Inference by LLMs

```bash
python prompt.py
```

### Evaluation

We first train and evaluate 3 GNN-based methods over identical test data.

For CARE-GNN:

```bash
cd GNN_Methods/CARE-GNN/
unzip /data/Amazon.zip && unzip /data/YelpChi.zip
python data_process.py
python train.py
```

Now we get the results of CARE-GNN and splited test set.

For PC-GNN:

```bash
cd ../PC-GNN/
unzip /data/Amazon.zip && unzip /data/YelpChi.zip
python src/data_process.py
python main.py --config ./config/pcgnn_yelpchi.yml
```

For GTAN:

```bash
cd ../antifraud/
unzip /data/Amazon.zip && unzip /data/YelpChi.zip
python feature_engineering/data_process.py
python main.py --method gtan
```

To generate the identical test dataset for LLMs' inference:

```bash
cd ../
python test_index2id.py
```



Then we need to evaluate the result of outputs of LLMs.

```bash
cd ../
```

For evaluating GPT-3.5-turbo's output:

```bash
python eval_gpt.py
```

For evaluating Llama 70B's output:

```bash
python eval_llama.py
```

To generate the result plot:

```bash
python plot.py
```

All the results are stored in `./output/`.

### Deploy vllm

This part is used for deploy llama-7B. In our paper, we actually used llama-70B deployed at other place.

#### Install vllm

```bash
cd vllm/vllm/
pip install -e .
```

#### Login to HuggingFace

```bash
huggingface-cli login
```

There will be a token needed. Copy and paste `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` to the terminal.

There will appear "Add token as git credential? (Y/n)". Enter `Y`.

#### Quick start

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

## Citation

```
@inproceedings{xiang2023semi,
  title={Semi-supervised credit card fraud detection via attribute-driven graph representation},
  author={Xiang, Sheng and Zhu, Mingzhi and Cheng, Dawei and Li, Enxia and Zhao, Ruihui and Ouyang, Yi and Chen, Ling and Zheng, Yefeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={12},
  pages={14557--14565},
  year={2023}
}
@inproceedings{liu2021pick,
  title={Pick and choose: a GNN-based imbalanced learning approach for fraud detection},
  author={Liu, Yang and Ao, Xiang and Qin, Zidi and Chi, Jianfeng and Feng, Jinghua and Yang, Hao and He, Qing},
  booktitle={Proceedings of the web conference 2021},
  pages={3168--3177},
  year={2021}
}
@inproceedings{dou2020enhancing,
  title={Enhancing graph neural network-based fraud detectors against camouflaged fraudsters},
  author={Dou, Yingtong and Liu, Zhiwei and Sun, Li and Deng, Yutong and Peng, Hao and Yu, Philip S},
  booktitle={Proceedings of the 29th ACM international conference on information \& knowledge management},
  pages={315--324},
  year={2020}
}
```

