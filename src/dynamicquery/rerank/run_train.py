import pandas as pd
import numpy as np
import argparse
import configparser
from functools import partial
import importlib
import os

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer
import torch.optim as optim

import train
from dynamicquery import utils
import extended_roberta_v1 as roberta_v1
import extended_roberta_v2 as roberta_v2
import extended_roberta_v3 as roberta_v3
import dataloaders

def run():
    # Handle CLI + Config retrieval
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path', type=str,
                        help='path where config lies')
    parser.add_argument('--no_tweet_emb', action='store_false')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

    # set appropriate path shortcuts
    cs_path = config["training"].get("candidate_selection_experiment")

    train_neg_path = os.path.join(cs_path, "negative_embs_train.npy")
    dev_neg_path = os.path.join(cs_path, "negative_embs_dev.npy")
    emb_path = os.path.join(cs_path, "claim_embs.npy")
    tweet_emb_path = os.path.join(cs_path, "tweet_embs.npy")


    # Load Model
    MAX_LENGTH = config["training"].getint("max_length")

    model_str = config["model"].get("model_string")
    if config["model"].getint("version") == 1:
        roberta = roberta_v1
    elif config["model"].getint("version") == 2:
        roberta = roberta_v2
    elif config["model"].getint("version") == 3:
        roberta = roberta_v3
    else:
        raise ValueError("model version not accepted")
    model = roberta.ExtendedRobertaForExternalClassification.from_pretrained(model_str)
    
    if config["training"].getboolean("pretrained"):
        model.load_state_dict(torch.load(os.path.join(args.experiment_path, "pretrained_model.pt")))
        print("Loaded Pretrained Model")
    
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenize = partial(tokenizer, **dict(
        truncation=True, 
        max_length=MAX_LENGTH, 
        padding="max_length", 
        return_attention_mask=True
    ))

    # load data
    neg_ids = np.load(train_neg_path)
    dev_neg_ids = np.load(dev_neg_path)
    neg_embs = np.load(emb_path)
    tweet_embs = np.load(tweet_emb_path, allow_pickle=True)

    tweets, test_tweets = utils.get_tweets()
    test_tweets = test_tweets[1:]
    train_conns, dev_conns, test_conns = utils.get_qrels()
    claims = utils.get_claims()
    
    BATCH_SIZE = config["training"].getint("batch_size")
    N_CANDIDATES = config["training"].getint("n_candidates", 5)

    train_dl = dataloaders.get_clef2021_reranked_dataloader(
        tokenize, 
        claims, 
        tweets, 
        train_conns,
        neg_embs,
        neg_ids[:,:N_CANDIDATES],
        tweet_embs[()]["train"],
        params={'batch_size':BATCH_SIZE, 'shuffle':True})

    dev_dl = dataloaders.get_clef2021_reranked_dataloader(
        tokenize, 
        claims, 
        tweets, 
        dev_conns,
        neg_embs,
        dev_neg_ids[:,:N_CANDIDATES],
        tweet_embs[()]["dev"],
        params={'batch_size':BATCH_SIZE, 'shuffle':False}) 

    # training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if config["training"].getint("adapter_epochs") > 0:
        optimizer = optim.AdamW(model.parameters(), lr=config["training"].getfloat("adapter_lr"))
        train.train(
            model, 
            optimizer, 
            device,
            train_dl,
            dev_dl,
            epochs=config["training"].getint("adapter_epochs"),
            print_steps=5,
            adapters_only=True, 
            cls_train=True,
            includes_tweet_state=args.no_tweet_emb,
            save_path=None
        )
    
    optimizer = optim.AdamW(model.parameters(), lr=config["training"].getfloat("lr"))
    train.train(
        model, 
        optimizer, 
        device,
        train_dl,
        dev_dl,
        epochs=config["training"].getint("epochs"),
        print_steps=5,
        adapters_only=config["training"].getboolean("adapters_only"), 
        cls_train=True,
        includes_tweet_state=args.no_tweet_emb,
        save_path=os.path.join(args.experiment_path, "trained_model.pt")
    )
    
if __name__ == '__main__':
    run()





