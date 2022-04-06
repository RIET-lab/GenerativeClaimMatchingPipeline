import pandas as pd
import json
import os
import numpy as np

DATAPATH = "data/train/"
TESTPATH = "data/subtask-2a--english"

def get_test_tweets():
    return pd.read_csv(os.path.join(TESTPATH, "tweets-test.tsv"), sep="\t", names=["id","tweet"])

def get_tweets():
    return pd.read_csv(os.path.join(DATAPATH, "tweets-train-dev.tsv"), sep="\t", names=["id","tweet"]), get_test_tweets()

def get_test_qrels():
    conn_names = ["tweet_id", "tweet_num", "claim_id", "claim_num"]
    test_conns = pd.read_csv(os.path.join(TESTPATH, "qrels-test.tsv"), sep="\t", names=conn_names)
    return test_conns

def get_qrels():
    conn_names = ["tweet_id", "tweet_num", "claim_id", "claim_num"]
    train_conns = pd.read_csv(os.path.join(DATAPATH, "qrels-train.tsv"), sep="\t", names=conn_names)
    dev_conns = pd.read_csv(os.path.join(DATAPATH, "qrels-dev.tsv"), sep="\t", names=conn_names)
    
    # get negatives
    top_negative_ranks = np.load("experiments/candidate_selection/shared_resources/train_negative_ranks.npy")
    train_conns["negative_claim_idx"] = top_negative_ranks

    dev_top_negative_ranks = np.load("experiments/candidate_selection/shared_resources/dev_negative_ranks.npy")
    dev_conns["negative_claim_idx"] = dev_top_negative_ranks
    
    return train_conns, dev_conns, get_test_qrels()

def get_claims():
    # claimpaths = [os.path.join(DATAPATH, "vclaims", f"{claim_id}.json") for claim_id in claim_ids]
    claimpath = os.path.join(DATAPATH, "vclaims")
    claimpaths = [os.path.join(claimpath, f) for f in os.listdir(claimpath)]
    def load_claim(path):
        with open(path) as f:
            return json.load(f)
    
    claims = [load_claim(path) for path in claimpaths]
    return pd.DataFrame(claims, columns=["title","subtitle","author","date","vclaim_id","vclaim"])