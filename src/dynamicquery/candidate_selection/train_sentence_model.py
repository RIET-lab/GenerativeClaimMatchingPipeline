import pandas as pd
import json
import os
import numpy as np
import argparse
import configparser
from sentence_transformers import SentenceTransformer
from functools import partial
import torch
import torch.optim as optim
nn = torch.nn

from dynamicquery import utils
from loss import mnr_loss
import dataloaders

def run():
    # Handle CLI + Config retrieval
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path', type=str,
                        help='path where config lies')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

    # Claim Data
    data = utils.load_data(config["data"].get("dataset"), negatives_path=config["data"].get("negatives_path"))
    train_queries, dev_queries = data["queries"]
    train_qrels, dev_qrels = data["qrels"]
    targets = data["targets"]

    # Model + Dataloaders
    model = SentenceTransformer(config["model"].get("model_string"))
    tokenize = partial(model.tokenizer, **dict(
        truncation=True, 
        max_length=config["training"].getint("max_length"), 
        padding="max_length", 
        return_attention_mask=True
    ))
    with_negatives = config["data"].get("negatives_path") is not None
    print(f"Training with negatives: {with_negatives}")
    train_dl = dataloaders.get_encoder_dataloader(tokenize, targets, train_queries, train_qrels, with_negatives=with_negatives,
                                                   params={'batch_size':config["training"].getint("batch_size"), 'shuffle':True})    
    dev_dl = dataloaders.get_encoder_dataloader(tokenize, targets, dev_queries, dev_qrels, with_negatives=with_negatives,
                                                   params={'batch_size':config["training"].getint("batch_size"), 'shuffle':False})

    optimizer = optim.AdamW(model.parameters(), lr=config["training"].getfloat("lr"))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Is using GPU: {torch.cuda.is_available()}")

    loss_fn = mnr_loss(temp=config["training"].getfloat("temperature"))

    PRINT_STEPS = 5
    model.to(device)
    for epoch in range(config["training"].getint("epochs")):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_dl, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = [elmt.to(device) for elmt in inputs]
            labels = [elmt.to(device) for elmt in labels]
            current_batch_size = inputs[0].shape[0]
            if (not with_negatives):
                inpt_dict = {
                    "input_ids":torch.cat([inputs[0], labels[0]]),
                    "attention_mask":torch.cat([inputs[1], labels[1]])
                }
            else:
                inpt_dict = {
                    "input_ids":torch.cat([inputs[0], labels[0], labels[2]]),
                    "attention_mask":torch.cat([inputs[1], labels[1], labels[3]])
                }
                
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inpt_dict)
            embeddings = outputs['sentence_embedding']
            loss = loss_fn(embeddings[:current_batch_size], embeddings[current_batch_size:])
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % PRINT_STEPS  == PRINT_STEPS-1:    # print every 2000 mini-batches
                print(f'TRAIN [{epoch + 1}, {i + 1:5d}] loss: {running_loss / PRINT_STEPS:.3f}')
                running_loss = 0.0

        running_loss = 0.0       
        for i, data in enumerate(dev_dl, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = [elmt.to(device) for elmt in inputs]
            labels = [elmt.to(device) for elmt in labels]
            current_batch_size = inputs[0].shape[0]
            inpt_dict = {
                "input_ids":torch.cat([inputs[0], labels[0]]),
                "attention_mask":torch.cat([inputs[1], labels[1]])
            }

            with torch.no_grad():
                outputs = model(inpt_dict)
                embeddings = outputs['sentence_embedding']
                loss = loss_fn(embeddings[:current_batch_size], embeddings[current_batch_size:])
                running_loss += (loss * embeddings.shape[0]).item()

        print(f'DEV [{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(dev_dl.dataset):.3f}')

    print('Finished Training')
    torch.save(model.state_dict(), os.path.join(args.experiment_path, "model.pt"))


if __name__ == '__main__':
    run()