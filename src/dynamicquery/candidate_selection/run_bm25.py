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
    parser.add_argument('dataset', type=str, default="clef2021-checkthat-task2a--english")
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()    
    
    sw_nltk = stopwords.words('english')
    porter = PorterStemmer()

    def preprocess(sentence):
        sentence = sentence.lower()
        sentence = re.sub(r'[^\w\s]', '', sentence)
        return [porter.stem(word) for word in sentence.split() if word not in sw_nltk]

    data = utils.load_data(args.dataset)
    train_queries, dev_queries = data["queries"]
    train_qrels, dev_qrels = data["qrels"]
    targets = data["targets"]
    
    corpus = utils.get_bm25_preprocess_fn(args.dataset)(targets)
    tokenized_corpus = [preprocess(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    def get_bm25_rankings(qrels, targets, queries):
        run_queries = queries.merge(qrels, left_on="query_id", right_on="query_id", how="inner")
        run_queries = run_queries.merge(targets, left_on="target_id", right_on="target_id", how="inner")
        run_queries = run_queries[["query", "target"]]
        target_idx = [targets.target.to_list().index(t_claim) for t_claim in run_queries.target.to_list()]

        queries = run_queries["query"].to_list()
        tokenized_queries = [preprocess(query) for query in queries]
        doc_scores = [bm25.get_scores(query) for query in tokenized_queries]
        doc_ranks = [score.argsort()[::-1] for score in doc_scores]
        label_ranks = [list(doc_rank).index(idx) for idx, doc_rank in zip(target_idx, doc_ranks)]

        return doc_scores, doc_ranks, target_idx, label_ranks
        
    def avg_prec(gold, rankings, n):
        is_rel = (np.array(rankings)[:n] == gold).astype(float)
        return (is_rel/np.arange(1,n+1)).sum()

    def mean_avg_prec(golds, rankings, n):
        avg_precs = [avg_prec(gold, rlist, n) for gold, rlist in zip(golds, rankings)]
        return np.array(avg_precs).mean()

    map_results = {}
    for ptn in ["train", "dev"]:
        if ptn == "train":
            doc_scores, doc_ranks, target_idx, label_ranks = get_bm25_rankings(train_qrels, targets, train_queries)

        elif ptn == "dev":
            doc_scores, doc_ranks, target_idx, label_ranks = get_bm25_rankings(dev_qrels, targets, dev_queries)
        # elif ptn == "test":
            # doc_scores, doc_ranks, claim_idx, label_ranks = get_bm25_rankings(test_qrels, claims, test_tweets)

        map_results[ptn] = []
        for n in [1,5,10,20]:
            map_results[ptn].append(mean_avg_prec(target_idx, doc_ranks, n))
            
        if args.save:
            save_dir = f"experiments/candidate_selection/bm25/{args.dataset}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            full_path = os.path.join(save_dir, f"ranks_{ptn}.npy")
            neg_path = os.path.join(save_dir, f"ranks_{ptn}_negatives.npy")
            top_negative_ranks = [doc_rank[~np.isin(doc_rank, target_idx)][0] for doc_rank in doc_ranks]
            np.save(neg_path, top_negative_ranks)
            np.save(full_path, doc_ranks)
            
    print("ptn [map@1, map@5, map@10, map@20]:\n", map_results) 
    
    
if __name__ == "__main__":
    run()