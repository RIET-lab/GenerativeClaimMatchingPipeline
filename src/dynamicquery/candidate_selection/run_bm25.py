import pandas as pd
import json
import os
from rank_bm25 import BM25Okapi
from matplotlib import pyplot as plt
import numpy as np
from dynamicquery import utils
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import argparse

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_negatives', action='store_true')
    args = parser.parse_args()    
    
    sw_nltk = stopwords.words('english')
    porter = PorterStemmer()

    def preprocess(sentence):
        sentence = sentence.lower()
        sentence = re.sub(r'[^\w\s]', '', sentence)
        return [porter.stem(word) for word in sentence.split() if word not in sw_nltk]

    tweets, test_tweets = utils.get_tweets()
    test_tweets = test_tweets[1:]
    train_conns, dev_conns, test_conns = utils.get_qrels(False)
    claims = utils.get_claims()
    
    corpus = claims[["title", "subtitle", "vclaim"]].apply(lambda x: x[0]+' '+x[1]+' '+x[2], axis=1).to_list()
    tokenized_corpus = [preprocess(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    def get_bm25_rankings(connections, claims, tweets):
        run_tweets = tweets.join(connections.set_index("tweet_id"), on="id", how="inner")
        run_tweets = run_tweets.join(claims.set_index("vclaim_id"), on="claim_id", how="inner")
        run_tweets = run_tweets[["tweet", "vclaim"]]
        claim_idx = [claims.vclaim.to_list().index(t_claim) for t_claim in run_tweets.vclaim.to_list()]

        queries = run_tweets.tweet.to_list()
        tokenized_queries = [preprocess(query) for query in queries]
        doc_scores = [bm25.get_scores(query) for query in tokenized_queries]
        doc_ranks = [score.argsort()[::-1] for score in doc_scores]
        label_ranks = [list(doc_rank).index(idx) for idx, doc_rank in zip(claim_idx, doc_ranks)]

        return doc_scores, doc_ranks, claim_idx, label_ranks
    
    # doc_scores, doc_ranks, claim_idx, label_ranks = get_bm25_rankings(train_conns, claims, tweets)
    
    def avg_prec(gold, rankings, n):
        is_rel = (np.array(rankings)[:n] == gold).astype(float)
        return (is_rel/np.arange(1,n+1)).sum()

    def mean_avg_prec(golds, rankings, n):
        avg_precs = [avg_prec(gold, rlist, n) for gold, rlist in zip(golds, rankings)]
        return np.array(avg_precs).mean()

    map_results = {}
    for ptn in ["train", "dev", "test"]:
        if ptn == "train":
            doc_scores, doc_ranks, claim_idx, label_ranks = get_bm25_rankings(train_conns, claims, tweets)
        elif ptn == "dev":
            doc_scores, doc_ranks, claim_idx, label_ranks = get_bm25_rankings(dev_conns, claims, tweets)
        elif ptn == "test":
            doc_scores, doc_ranks, claim_idx, label_ranks = get_bm25_rankings(test_conns, claims, test_tweets)

        map_results[ptn] = []
        for n in [1,5,10,20]:
            map_results[ptn].append(mean_avg_prec(claim_idx, doc_ranks, n))
            
        if args.save_negatives:
            full_path = "./experiments/candidate_selection/shared_resources/ranks_{}.npy".format(ptn)
            neg_path = "./experiments/candidate_selection/shared_resources/{}_negative_ranks.npy".format(ptn)
            print("saving a negative rank list to {}".format("./experiments/candidate_selection/shared_resources"))
            top_negative_ranks = [doc_rank[~np.isin(doc_rank, claim_idx)][0] for doc_rank in doc_ranks]
            np.save(neg_path, top_negative_ranks)
            np.save(full_path, doc_ranks)
            
    print("ptn [map@1, map@5, map@10, map@20]:\n", map_results) 
    
    
if __name__ == "__main__":
    run()