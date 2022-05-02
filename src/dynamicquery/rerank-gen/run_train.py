import argparse
import configparser
import os
from functools import partial
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoModel
from transformers import AutoTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments

from dynamicquery import utils
from models.MyGPTNeo import MyGPTNeoForCausalLM
from dataloaders import ExtendedAutoRegressiveDataset
import trainers

nn = torch.nn

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, 
                        help="path where config.ini lies")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

    # Load Model
    model_str = config["model"].get("model_string")

    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token

    model = MyGPTNeoForCausalLM.from_pretrained(model_str)
    # AutoModelForCausalLM.from_pretrained(model_str)

    # Setup Data
    tweets, test_tweets = utils.get_tweets()
    test_tweets = test_tweets[1:]
    train_conns, dev_conns, test_conns = utils.get_qrels()
    claims = utils.get_claims()

    cs_path = config["training"].get("candidate_selection", None)
    if cs_path:
        print(f"Getting candidates from {cs_path}")
        train_neg_path = os.path.join(cs_path, "negative_embs_train.npy")
        dev_neg_path = os.path.join(cs_path, "negative_embs_dev.npy")
        neg_ids = np.load(train_neg_path)
        dev_neg_ids = np.load(dev_neg_path)
    else:
        neg_ids = None
        dev_neg_ids = None

    # setup tokenize fn
    tokenizer.pad_token = tokenizer.eos_token
    def _tokenize(text, tokenizer, max_length, prefix=""):
        text = tokenizer.bos_token + prefix + text + tokenizer.eos_token
        token_params = dict(
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True
        )
        return tokenizer(text, **token_params)

    max_length=config["training"].getint("max_length")
    tweet_tokenize = partial(_tokenize, tokenizer=tokenizer, max_length=max_length, prefix="tweet:")
    claim_tokenize = partial(_tokenize, tokenizer=tokenizer, max_length=max_length, prefix="claim:")

    optimization = config["training"].get("optimization", None)
    mask_prior = optimization in ["nl3u", "mle", "rll"]
    # mask_prior = False
    print(f"Masking prior: {mask_prior}")

    train_dataset = ExtendedAutoRegressiveDataset(
        tweet_encode_fn=tweet_tokenize, 
        claim_encode_fn=claim_tokenize,
        claims=claims, 
        tweets=tweets, 
        connections=train_conns,
        mask_prior=mask_prior,
        ranks=neg_ids,
        prior_prob=config["training"].getfloat("prior_prob", 0.0),
        include_posterior=optimization=="mutual_information",
        training=optimization,
    )

    dev_dataset = ExtendedAutoRegressiveDataset(
        tweet_encode_fn=tweet_tokenize, 
        claim_encode_fn=claim_tokenize,
        claims=claims, 
        tweets=tweets, 
        connections=dev_conns,
        mask_prior=mask_prior,
        ranks=dev_neg_ids,
        include_posterior=optimization=="mutual_information",
        training=optimization,
    )

    # Setup Training
    training_args = TrainingArguments(
        output_dir=os.path.join(args.experiment_path, config["training"].get("save_dir")),
        num_train_epochs=config["training"].getint("epochs"),
        learning_rate=config["training"].getfloat("lr"),
        per_device_train_batch_size=config["training"].getint("per_chip_batch_size"),
        per_device_eval_batch_size=config["training"].getint("per_chip_batch_size"),
        evaluation_strategy="steps",
        eval_steps=config["training"].getint("eval_steps"),
        logging_steps=config["training"].getint("print_steps"),
        lr_scheduler_type=config["training"].get("lr_schedule", "constant"),
        save_strategy='epoch')

    flip = config["training"].getboolean("flip")

    if optimization == "mixed":
        trainer_cls = partial(trainers.MixedTrainer, max_length=max_length)
    elif optimization == "mutual_information":
        trainer_cls = partial(trainers.HingedMutualInformationTrainer, max_length=max_length)
    elif optimization == "nl3u":
        trainer_cls = trainers.NL3UTrainer
    elif optimization == "rll":
        trainer_cls = partial(trainers.RLLTrainer, loss_constant=config["training"].getfloat("rll_constant", 1e-1))
    else:
        trainer_cls = Trainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        # compute_metrics=compute_metrics,
    )

    trainer.train()



if __name__ == "__main__":
    run()