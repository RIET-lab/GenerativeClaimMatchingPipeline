# Setup
Install dependancies from requirements file (currently this is heavy, in future planning to skim some of this)  
`pip install -r requirements.txt`

Setup utilities/package with pip  
`python setup.py install`

Define aliases for quick script running  
`source aliases.sh`

Download stopwords from nltk for bm25  
`python -c "import nltk; nltk.download('stopwords')"`


# Data
We used this pipeline of CLEF dataset. To download, run  
`./download_clef.sh`


# Candidate Selection
Sentence Transformer is the first stage of the claim matching pipeline. We have set it up to work with Huggingfaces implementations of sentence T5 (for all corresponding sizes). We also experiment with BM25 as a baseline done in previous works. We also use BM25 to retrieve negatives for training the sentence model

### Setup Experimental Config
create a config.ini that looks like
```
[training]
finetune = True
lr = 5e-6
batch_size = 6
max_length = 128
temperature = .1
epochs = 1
with_negatives = True

[model]
model_string = sentence-transformers/sentence-t5-large
```
Most attributes are self explanatory, except with_negatives is a boolean referring to if loss is includes hard negatives. If so they must be ranked and stored in a `negative_embs_<partition>.npy` file

### Getting BM25 Negatives

Only needed if above config includes `with_negative = True`. Though heavily reccomended, since otherwise problem becomes to easy and doesnt generalize well. Use the following command:  
```
selection-bm25 --save_negatives
```
It will default to saving to experiments/candidate_selection/shared_resources. Note you only need to do this once, all experiments will draw from there.

### Run the Candidate Selection Training
```
selection-train <path to where experiment config.ini exists>
```

### Evaluating of Candidate
You can evaluate different experiments after training using  
```
selection-eval <path to experiment config.ini>
```

Note that if you want to evaluate the baseline model without the trained weights and just use its pretrained weights you can do so by adding a `--raw` flag

Additionally in case when you want to use these candidates for reranking (next stage of pipeline), youll need to also save out the embeddings and rankings. You can do this via `--save_embs` and `--save_ranks` flags. So the call for reranking setup would be:  
```
selection-eval <path to experiment config.ini> --save_embs --save_ranks
```
Note on this call dont use raw flag since otherwise itll save untrained embeddings, unless you intend to rerank based off of that.


# Reranking Model
Training the Cross Attention/Query Model requires initial pretraining to align the new parameters to the embedding space of the candidates

### Setup Experimental Config
create a config.ini that looks like
```
[pretraining]
lr = 1e-4
batch_size = 32
adapters_only = True
max_length = 192
epochs = 10
with_negatives = False
candidate_selection_experiment = ./experiments/candidate_selection/finetune_st5_large_claims_negs

[training]
lr = 2e-5
batch_size = 32
adapters_only = False
max_length = 192
epochs = 20
candidate_selection_experiment = ./experiments/candidate_selection/finetune_st5_large_claims_negs
pretrained = True

[model]
model_string = roberta-base
version = 1
```
Most attributes are self explanatory, except with_negatives is a boolean referring to if loss is includes hard negatives. If so they must be ranked and stored in a `negative_embs_<partition>.npy` file

### Run Reranking Model Pretraining
```
rerank-pretrain <path to where experiment config.ini exists>
```
an example:
```
rerank-pretrain experiments/cross_query/base_rndm_large_neg_v1/
```


### Train Reranking Model
```
rerank-train <path to where experiment config.ini exists>
```
an example:
```
rerank-train experiments/cross_query/base_rndm_large_neg_v1/
```

### Evaluation of Rerankings
```
rerank-eval <path to where experiment config.ini exists>
```
an example:
```
rerank-eval experiments/cross_query/base_rndm_large_neg_v1/
```