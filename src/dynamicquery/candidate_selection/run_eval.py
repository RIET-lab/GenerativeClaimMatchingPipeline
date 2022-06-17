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
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save-embs', action='store_true')
    parser.add_argument('--save-ranks', action='store_true')
    parser.add_argument('--queries-path', type=str, default=None)
    parser.add_argument('--targets-path', type=str, default=None)
    parser.add_argument('--targets-names', nargs='+', default=[])
    parser.add_argument('--qrels-path', type=str, default=None)
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

    assert args.queries_path is None or args.save_ranks, "can only overwrite query path if were just ranking, since we will lack labels"

    # Collect data
    # tweets, test_tweets = utils.get_tweets()
    # test_tweets = test_tweets[1:]
    # train_conns, dev_conns, test_conns = utils.get_qrels()

    data = utils.load_data(
        config["data"].get("dataset"), 
        negatives_path=config["data"].get("negatives_path"),
        test_data=args.test    
    )
    train_queries, dev_queries = data["queries"]
    train_qrels, dev_qrels = data["qrels"]
    targets = data["targets"]
    if args.test:
        test_queries, test_qrels = data["test"]
    
    if args.targets_path is not None:
        print(f"getting targets from {args.targets_path}")
        targets = utils.get_targets(args.targets_path, args.targets_names)

    # setup model
    model_str = config["model"].get("model_string")
    ft_str = os.path.join(args.experiment_path, "model.pt")
    model = SentenceTransformer(model_str)

    if not args.raw:
        model.load_state_dict(torch.load(ft_str))

    # get embeddings of all claoims
    embs = model.encode(targets.target.to_list())

    # eval fns
    def get_idx(connections, claims, tweets):
        run_tweets = tweets.merge(connections, on="query_id", how="inner")
        run_tweets = run_tweets.merge(targets, on="target_id", how="inner")
        run_tweets = run_tweets[["query", "target"]].reset_index()
        claim_idx = [targets.target.to_list().index(t_claim) for t_claim in run_tweets.target.to_list()]
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

    if args.queries_path is None:
        save_path = args.experiment_path 
        partitions = ["train", "dev"]
        if args.test: partitions.append("test")
    else:
        save_path = os.path.join(args.experiment_path, "custom_queries")
        partitions = ["custom"]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    print(f"Saving files to {save_path}")

    map_results = {}
    map_recall_results = {}
    all_tweet_embs = {}
    for ptn in partitions:
        if ptn == "train":
            run_tweets, claim_idx = get_idx(train_qrels, targets, train_queries)
        elif ptn == "dev":
            run_tweets, claim_idx = get_idx(dev_qrels, targets, dev_queries)
        elif ptn == "test":
            run_tweets, claim_idx = get_idx(test_qrels, targets, test_queries)
        elif ptn == "custom":
            run_tweets = utils.get_queries(args.queries_path)
            if args.qrels_path is not None:
                custom_qrels = utils.get_qrels(args.qrels_path)
                run_tweets, claim_idx = get_idx(custom_qrels, targets, run_tweets)


        tweet_embs = model.encode(run_tweets['query'].to_list())
        all_tweet_embs[ptn] = tweet_embs
        scores = tweet_embs @ embs.T
        ranks = [score.argsort()[::-1] for score in scores]
        if args.save_ranks:
            np.save(os.path.join(save_path, f"ranks_{ptn}.npy"),
                    np.array(ranks))
            np.save(os.path.join(save_path, f"ranks_{ptn}_negatives.npy"),
                    get_negative_ranks_arr(ranks, claim_idx))

        if ptn != "custom" or args.qrels_path is not None:
            map_results[ptn] = []
            for n in [1,5,10,20]:
                map_results[ptn].append(mean_avg_prec(claim_idx, ranks, n))

            map_recall_results[ptn] = []
            for n in [1,5,10,20]:
                map_recall_results[ptn].append(mean_recall(claim_idx, ranks, n))
                    
    if args.save_embs:
        np.save(os.path.join(save_path, "tweet_embs.npy"), all_tweet_embs)
        np.save(os.path.join(save_path, "claim_embs.npy"), embs)

    # print eval metrics
    print("ptn [map@1, map@5, map@10, map@20]:\n", map_results)
    print()
    print("ptn [rec@1, rec@5, rec@10, rec@20]:\n", map_recall_results)

if __name__ == "__main__":
    run()