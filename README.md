# Setup
Install dependancies from requirements file (currently this is heavy, in future planning to skim some of this)  
`pip install -r requirements.txt`

Setup utilities/package with pip  
`pip install .`

Define aliases for quick script running  
`source aliases.sh`

CLEF data should be downloaded and unzipped into `data/`


# Candidate Selection
Sentence Transformer is the first stage of the claim matching pipeline. We have set it up to work with Huggingfaces implementations of sentence T5 (for all corresponding sizes)

### Setup experimental config
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

### Run the Candidate Selection Training
```
selection-train <path to where experiment config.ini exists>
```

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