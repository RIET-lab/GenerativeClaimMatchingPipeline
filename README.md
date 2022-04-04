## Download Requirements
`pip install -r requirements.txt`

## Train Sentence Transformer
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

### Run the training
```
python src/dynamicquery/train_sentence_model.py <path to where experiment config.ini exists>
```

## Train Cross Attention