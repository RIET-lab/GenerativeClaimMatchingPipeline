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
import extended_roberta as roberta
import dataloaders

def run():
    # Handle CLI + Config retrieval
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path', type=str,
                        help='path where config lies')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

    # set appropriate path shortcuts
    cs_path = config["pretraining"].get("candidate_selection_experiment")

    train_neg_path = os.path.join(cs_path, "negative_embs_train.npy")
    dev_neg_path = os.path.join(cs_path, "negative_embs_dev.npy")
    emb_path = os.path.join(cs_path, "claim_embs.npy")


    # Load Model
    MAX_LENGTH = config["pretraining"].getint("max_length")

    model_str = config["model"].get("model_string")
    model = roberta.ExtendedRobertaForExternalClassification.from_pretrained(model_str)
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
    claims = utils.get_claims()

    BATCH_SIZE = config["pretraining"].getint("batch_size")

    train_dl = dataloaders.get_clef2021_pretraining_dataloader(
        tokenize, 
        claims, 
        neg_embs,
        n_negatives=5,
        params={'batch_size':BATCH_SIZE, 'shuffle':True})

    # dev_dl = dataloaders.get_clef2021_pretraining_dataloader(
    #     tokenize, 
    #     claims, 
    #     neg_embs,
    #     n_negatives=5,
    #     params={'batch_size':BATCH_SIZE, 'shuffle':False}) 

    # training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.AdamW(model.parameters(), lr=config["pretraining"].getfloat("lr"))

    train.train(
        model, 
        optimizer, 
        device,
        train_dl,
        None,
        epochs=config["pretraining"].getint("epochs"),
        print_steps=5,
        adapters_only=config["pretraining"].getboolean("adapters_only"), 
        cls_train=True,
        save_path=os.path.join(args.experiment_path, "pretrained_model.pt")
    )
    
if __name__ == '__main__':
    run()





