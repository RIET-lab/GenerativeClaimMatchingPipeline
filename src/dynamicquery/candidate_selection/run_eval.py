import pandas as pd
import json
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import argparse
import configparser

from dynamicquery import utils

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path', type=str,
                        help='path where config lies')
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--save_embs', action='store_true')
    parser.add_argument('--save_ranks', action='store_true')
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

    # Collect data
    tweets, test_tweets = utils.get_tweets()
    test_tweets = test_tweets[1:]
    train_conns, dev_conns, test_conns = utils.get_qrels()
    claims = utils.get_claims()

    # setup model
    model_str = config["model"].get("model_string")
    ft_str = os.path.join(args.experiment_path, "model.pt")
    model = SentenceTransformer(model_str)

    if not args.raw:
        model.load_state_dict(torch.load(ft_str))

    # get embeddings of all claoims
    embs = model.encode(claims.vclaim.to_list())

    # eval fns
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
    all_tweet_embs = {}
    for ptn in ["train", "dev", "test"]:
        if ptn == "train":
            run_tweets, claim_idx = get_idx(train_conns, claims, tweets)
        elif ptn == "dev":
            run_tweets, claim_idx = get_idx(dev_conns, claims, tweets)
        elif ptn == "test":
            run_tweets, claim_idx = get_idx(test_conns, claims, test_tweets)

        tweet_embs = model.encode(run_tweets.tweet.to_list())
        all_tweet_embs[ptn] = tweet_embs
        scores = tweet_embs @ embs.T
        ranks = [score.argsort()[::-1] for score in scores]
        if args.save_ranks:
            np.save(os.path.join(args.experiment_path, f"ranks_{ptn}.npy"),
                    np.array(ranks))
        
        if args.save_embs:
            np.save(os.path.join(args.experiment_path, f"negative_embs_{ptn}.npy"),
                    get_negative_ranks_arr(ranks, claim_idx))

        map_results[ptn] = []
        for n in [1,5,10,20]:
            map_results[ptn].append(mean_avg_prec(claim_idx, ranks, n))

        map_recall_results[ptn] = []
        for n in [1,5,10,20]:
            map_recall_results[ptn].append(mean_recall(claim_idx, ranks, n))
                    
    if args.save_embs:
        np.save(os.path.join(args.experiment_path, "tweet_embs.npy"), all_tweet_embs)
        np.save(os.path.join(args.experiment_path, "claim_embs.npy"), embs)

    # print eval metrics
    print("ptn [map@1, map@5, map@10, map@20]:\n", map_results)
    print()
    print("ptn [rec@1, rec@5, rec@10, rec@20]:\n", map_recall_results)

if __name__ == "__main__":
    run()