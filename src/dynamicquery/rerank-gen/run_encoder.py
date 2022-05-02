import numpy as np
from functools import partial
from tqdm import tqdm
import argparse
import configparser
import os

import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer

from dynamicquery import utils
from dataloaders import get_encoding_dataloader

def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def run():
    # Handle CLI + Config retrieval
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path', type=str,
                        help='path where config lies')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

    # load model
    model_str = config["model"].get("model_string")
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    model = AutoModel.from_pretrained(model_str)

    # setup tokenize fn
    tokenizer.pad_token = tokenizer.eos_token
    def _tokenize(text, tokenizer, max_length):
        text = tokenizer.bos_token + text + tokenizer.eos_token
        token_params = dict(
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True
        )
        return tokenizer(text, **token_params)

    tokenize = partial(_tokenize, tokenizer=tokenizer, max_length=config["encoder"].getint("max_length"))

    # load data
    claims = utils.get_claims()
    tokenized_claims = claims.vclaim.apply(tokenize).tolist()
    dataloader = get_encoding_dataloader(tokenized_claims, dict(
        batch_size=config["encoder"].getint("batch_size"), 
        shuffle=False))

    # encode
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    claim_path = os.path.join(args.experiment_path, "encoded_claims")
    safe_mkdir(claim_path)

    model.eval()
    outpts = []
    idx, preidx, idx_list = 0, 0, [0]
    checkpoint_steps = config["encoder"].getint("checkpoint_steps")
    for input_ids, attention_mask in tqdm(dataloader):
        idx += input_ids.shape[0]
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outpt = model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        outpt = list(map(lambda x: x.cpu().detach().numpy(), outpt.hidden_states))
        outpts.append(outpt)
        if idx - preidx >= checkpoint_steps or idx >= len(dataloader.dataset):
            idx_list.append(idx)
            outpts = [np.concatenate(x, axis=0) for x in list(zip(*outpts))]
            np.save(os.path.join(claim_path, f"encoded_claims_{preidx}_{idx}.npy"), outpts)
            preidx = idx
            outpts = []

    np.save(os.path.join(claim_path, "idx_bins.npy"), np.array(idx_list))


if __name__ == "__main__":
    run()