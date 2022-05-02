import os
import pandas as pd
from functools import partial
import numpy as np
import argparse
import configparser

from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from transformers import AutoTokenizer

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
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no_tweet_emb', action='store_false')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))
    
    # set appropriate path shortcuts
    cs_path = config["training"].get("candidate_selection_experiment")

    train_neg_path = os.path.join(cs_path, "ranks_train.npy")
    dev_neg_path = os.path.join(cs_path, "ranks_dev.npy")
    test_neg_path = os.path.join(cs_path, "ranks_test.npy")
    emb_path = os.path.join(cs_path, "claim_embs.npy")
    tweet_emb_path = os.path.join(cs_path, "tweet_embs.npy")
    
    neg_ids = np.load(train_neg_path)
    dev_neg_ids = np.load(dev_neg_path)
    test_neg_ids = np.load(test_neg_path)
    neg_embs = np.load(emb_path)
    tweet_embs = np.load(tweet_emb_path, allow_pickle=True)

    MAX_LENGTH = config["training"].getint("max_length")

    # Load Model
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
    model.load_state_dict(torch.load(os.path.join(args.experiment_path, "trained_model.pt")))
    print("loaded model from {}".format(os.path.join(args.experiment_path, "trained_model.pt")))
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenize = partial(tokenizer, **dict(
        truncation=True, 
        max_length=MAX_LENGTH, 
        padding="max_length", 
        return_attention_mask=True
    ))
    
    # Claim Data
    tweets, test_tweets = utils.get_tweets()
    test_tweets = test_tweets[1:]
    train_conns, dev_conns, test_conns = utils.get_qrels()
    claims = utils.get_claims()

    BATCH_SIZE = 32
    N_CANDIDATES = config["training"].getint("n_candidates", 5)
    
    train_dl = dataloaders.get_clef2021_reranked_eval_dataloader(
        tokenize, 
        claims, 
        tweets, 
        train_conns,
        neg_embs,
        neg_ids[:,:N_CANDIDATES],
        tweet_embeddings=tweet_embs[()]["train"] if args.no_tweet_emb else None,
        params={'batch_size':BATCH_SIZE, 'shuffle':False})

    dev_dl = dataloaders.get_clef2021_reranked_eval_dataloader(
        tokenize, 
        claims, 
        tweets, 
        dev_conns,
        neg_embs,
        dev_neg_ids[:,:N_CANDIDATES],
        tweet_embeddings=tweet_embs[()]["dev"] if args.no_tweet_emb else None,
        params={'batch_size':BATCH_SIZE, 'shuffle':False}) 
    
    # Run
    model.eval()
    
    # train ptn
    probits = []
    for inputs, external_inputs in train_dl:
        inpt_dict = {
            "input_ids": inputs[0],
            "attention_mask": inputs[1],
            "extended_states": external_inputs,
            "includes_tweet_state": args.no_tweet_emb,
        }
        with torch.no_grad():
            out = model(**inpt_dict)
            _probits = torch.nn.functional.softmax(out.logits[:,:], dim=-1)
            # print(_probits.shape, external_inputs.shape)
        probits.append(_probits.detach().numpy())
    probits = np.concatenate(probits, 0)
    reranks = probits.argsort()[:,::-1]
    
    # dev ptn
    dev_probits = []
    for inputs, external_inputs in dev_dl:
        inpt_dict = {
            "input_ids": inputs[0],
            "attention_mask": inputs[1],
            "extended_states": external_inputs,
            "includes_tweet_state": args.no_tweet_emb,
        }
        with torch.no_grad():
            out = model(**inpt_dict)
            _probits = torch.nn.functional.softmax(out.logits[:,:], dim=-1)
        dev_probits.append(_probits.detach().numpy())
    dev_probits = np.concatenate(dev_probits, 0)    
    dev_reranks = dev_probits.argsort()[:,::-1]
    
    # test ptn
    if args.test:
        test_dl = dataloaders.get_clef2021_reranked_eval_dataloader(
            tokenize, 
            claims, 
            test_tweets, 
            test_conns,
            neg_embs,
            test_neg_ids[:,:N_CANDIDATES],
            tweet_embeddings=tweet_embs[()]["test"] if args.no_tweet_emb else None,
            params={'batch_size':BATCH_SIZE, 'shuffle':False}) 
        
        test_probits = []
        for inputs, external_inputs in test_dl:
            inpt_dict = {
                "input_ids": inputs[0],
                "attention_mask": inputs[1],
                "extended_states": external_inputs,
                "includes_tweet_state": args.no_tweet_emb,
            }
            with torch.no_grad():
                out = model(**inpt_dict)
                _probits = torch.nn.functional.softmax(out.logits[:,:], dim=-1)
                # _probits = torch.einsum('...ld,...d->...l', external_inputs[:,:-1], external_inputs[:,-1])
            test_probits.append(_probits.detach().numpy())
        test_probits = np.concatenate(test_probits, 0)    
        test_reranks = test_probits.argsort()[:,::-1]
    
    def get_idx(connections, claims, tweets):
        run_tweets = tweets.join(connections.set_index("tweet_id"), on="id", how="inner")
        run_tweets = run_tweets.join(claims.set_index("vclaim_id"), on="claim_id", how="inner")
        run_tweets = run_tweets[["tweet", "vclaim"]].reset_index()
        claim_idx = [claims.vclaim.to_list().index(t_claim) for t_claim in run_tweets.vclaim.to_list()]
        return run_tweets, claim_idx

    def avg_prec(gold, rankings, n):
        is_rel = (np.array(rankings)[:n] == gold).astype(float)
        return (is_rel/np.arange(1,n+1)).sum()

    def recall(gold, rankings, n):
        is_rel = (np.array(rankings)[:n] == gold).astype(float)
        return is_rel.sum()

    def mean_avg_prec(golds, rankings, n):
        avg_precs = [avg_prec(gold, rlist, n) for gold, rlist in zip(golds, rankings)]
        return np.array(avg_precs).mean()

    def mean_recall(golds, rankings, n):
        avg_precs = [recall(gold, rlist, n) for gold, rlist in zip(golds, rankings)]
        return np.array(avg_precs).mean()

    def get_negative_ranks(ranks, gold):
        return [r for r in ranks if r!=gold]

    def get_negative_ranks_arr(ranks, gold):
        n_ranks = [get_negative_ranks(r, g) for r,g in zip(ranks, claim_idx)]
        return np.array(n_ranks)

    map_results = {}
    map_recall_results = {}
    partitions = ["train", "dev"]
    if args.test: partitions.append("test")
    k_values = list(filter(lambda k: k<=N_CANDIDATES, [1,5,10,20]))
    for ptn in partitions:
        if ptn == "train":
            run_tweets, claim_idx = get_idx(train_conns, claims, tweets)
            ranks = np.array([ids[rerank] for ids, rerank in zip(neg_ids, reranks)])
        elif ptn == "dev":
            run_tweets, claim_idx = get_idx(dev_conns, claims, tweets)
            ranks = np.array([ids[rerank] for ids, rerank in zip(dev_neg_ids, dev_reranks)])
        elif ptn == "test":
            run_tweets, claim_idx = get_idx(test_conns, claims, test_tweets)
            ranks = np.array([ids[rerank] for ids, rerank in zip(test_neg_ids, test_reranks)])

        map_results[ptn] = []
        for n in k_values:
            map_results[ptn].append(mean_avg_prec(claim_idx, ranks, n))

        map_recall_results[ptn] = []
        for n in k_values:
            map_recall_results[ptn].append(mean_recall(claim_idx, ranks, n))
      
    map_strings = ", ".join(["map@{}".format(k) for k in k_values])
    print("ptn [{}]:\n".format(map_strings), map_results)
    print()
    rec_strings = ", ".join(["rec@{}".format(k) for k in k_values])
    print("ptn [{}]:\n".format(rec_strings), map_recall_results)
    
if __name__ == '__main__':
    run()
    
    
    