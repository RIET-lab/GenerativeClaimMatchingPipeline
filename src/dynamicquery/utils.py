import pandas as pd
import json
import os
import numpy as np
from dynamicquery import defaults

DATAPATH = "data/train/"
TESTPATH = "data/subtask-2a--english"

#############

def get_queries(querypath):
    return pd.read_csv(querypath, sep="\t", names=["query_id", "query"]).drop_duplicates()

def get_qrels(qrelpath):
    conn_names = ["query_id", "query_num", "target_id", "target_num"]
    return pd.read_csv(qrelpath, sep="\t", names=conn_names).drop_duplicates()

def get_targets(targetpath, keynames):
    targetpaths = [os.path.join(targetpath, f) for f in os.listdir(targetpath)]
    def load_claim(path):
        with open(path) as f:
            return json.load(f)
    targets = [load_claim(path) for path in targetpaths]
    df = pd.DataFrame(targets)
    df.columns = keynames
    return df.drop_duplicates()

def load_data(dataset, negatives_path=None, test_data=False):
    """Returns data in pandas dataframes
    Args
        dataset: name of the task
        negatives_path: path to where ranks_{ptn}_negative.npy lies
    """
    if dataset == "clef2021-checkthat-task2a--english":
        train_queries = get_queries(defaults.TASK_2A_EN_TRAIN_QUERY_PATH)
        dev_queries = get_queries(defaults.TASK_2A_EN_DEV_QUERY_PATH)

        train_qrels = get_qrels(defaults.TASK_2A_EN_TRAIN_QREL_PATH)
        dev_qrels = get_qrels(defaults.TASK_2A_EN_DEV_QREL_PATH)

        targets = get_targets(defaults.TASK_2A_EN_TARGETS_PATH, defaults.TASK_2A_EN_TARGETS_KEY_NAMES)

        if negatives_path is not None:
            top_negative_ranks = np.load(os.path.join(negatives_path, "ranks_train_negatives.npy"))
            train_qrels["negative_target_idx"] = top_negative_ranks

            dev_top_negative_ranks = np.load(os.path.join(negatives_path, "ranks_dev_negatives.npy"))
            dev_qrels["negative_target_idx"] = dev_top_negative_ranks

        if test_data:
            test_queries = get_queries(defaults.TASK_2A_EN_TEST21_QUERY_PATH)
            test_qrels = get_qrels(defaults.TASK_2A_EN_TEST21_QREL_PATH)
            test_data = (test_queries, test_qrels)
        else:
            test_data = None

        return dict(
            queries = (train_queries, dev_queries),
            qrels = (train_qrels, dev_qrels),
            targets = targets,
            test = test_data
        )

    elif dataset == "clef2022-checkthat-task2b--english":
        train_queries = get_queries(defaults.TASK_2B_EN_TRAIN_QUERY_PATH)
        dev_queries = get_queries(defaults.TASK_2B_EN_DEV_QUERY_PATH)

        train_qrels = get_qrels(defaults.TASK_2B_EN_TRAIN_QREL_PATH)
        dev_qrels = get_qrels(defaults.TASK_2B_EN_DEV_QREL_PATH)

        targets = get_targets(defaults.TASK_2B_EN_TARGETS_PATH, defaults.TASK_2B_EN_TARGETS_KEY_NAMES)

        if negatives_path is not None:
            top_negative_ranks = np.load(os.path.join(negatives_path, "ranks_train_negatives.npy"))
            train_qrels["negative_target_idx"] = top_negative_ranks

            dev_top_negative_ranks = np.load(os.path.join(negatives_path, "ranks_dev_negatives.npy"))
            dev_qrels["negative_target_idx"] = dev_top_negative_ranks

        if test_data:
            test_queries = get_queries(defaults.TASK_2B_EN_TEST21_QUERY_PATH)
            test_qrels = get_qrels(defaults.TASK_2B_EN_TEST21_QREL_PATH)
            test_data = (test_queries, test_qrels)
        else:
            test_data = None
        

        return dict(
            queries = (train_queries, dev_queries),
            qrels = (train_qrels, dev_qrels),
            targets = targets,
            test = test_data
        )

    elif dataset == "clef2022-checkthat-task2a--arabic":
        def fix_queries(df):
            df = df[1:]
            df.columns = ["_", "query"]
            df.reset_index(inplace=True)
            df = df.rename(columns = {"index": "query_id"})
            return df[["query_id", "query"]]
                
        train_queries = fix_queries(get_queries(defaults.TASK_2A_AR_TRAIN_QUERY_PATH))
        dev_queries = fix_queries(get_queries(defaults.TASK_2A_AR_DEV_QUERY_PATH))

        train_qrels = get_qrels(defaults.TASK_2A_AR_TRAIN_QREL_PATH)
        dev_qrels = get_qrels(defaults.TASK_2A_AR_DEV_QREL_PATH)

        targets = get_targets(defaults.TASK_2A_AR_TARGETS_PATH, defaults.TASK_2A_AR_TARGETS_KEY_NAMES)

        if negatives_path is not None:
            top_negative_ranks = np.load(os.path.join(negatives_path, "ranks_train_negatives.npy"))
            train_qrels["negative_target_idx"] = top_negative_ranks

            dev_top_negative_ranks = np.load(os.path.join(negatives_path, "ranks_dev_negatives.npy"))
            dev_qrels["negative_target_idx"] = dev_top_negative_ranks

        if test_data:
            test_queries = fix_queries(get_queries(defaults.TASK_2A_AR_TEST21_QUERY_PATH))
            test_qrels = get_qrels(defaults.TASK_2A_AR_TEST21_QREL_PATH)
            test_data = (test_queries, test_qrels)
        else:
            _train_queries = get_queries(defaults.TASK_2A_EN_TRAIN_QUERY_PATH)
            train_queries = pd.concat([train_queries, _train_queries], ignore_index=True, sort=False)

            _train_qrels = get_qrels(defaults.TASK_2A_EN_TRAIN_QREL_PATH)
            train_qrels = pd.concat([train_qrels[["target_id", "target", "title"]], 
                _train_qrels[["target_id", "target", "title"]]], ignore_index=True, sort=False)

            _targets = get_targets(defaults.TASK_2A_EN_TARGETS_PATH, defaults.TASK_2A_EN_TARGETS_KEY_NAMES)
            targets = pd.concat([train_qrels, _train_qrels], ignore_index=True, sort=False)

            test_data = None

        return dict(
            queries = (train_queries, dev_queries),
            qrels = (train_qrels, dev_qrels),
            targets = targets,
            test = test_data
        )

    else:
        raise ValueError(f"{dataset} isnt a default. please include paths manually")

def get_bm25_preprocess_fn(dataset):
    if dataset == "clef2021-checkthat-task2a--english":
        return lambda targets: targets[["title", "subtitle", "target"]].apply(lambda x: x[0]+' '+x[1]+' '+x[2], axis=1).to_list()
    
    elif dataset == "clef2022-checkthat-task2b--english":
        return lambda targets: targets[["title", "target"]].apply(lambda x: x[0]+' '+x[1], axis=1).to_list()

    elif dataset == "clef2022-checkthat-task2a--arabic":
        return lambda targets: targets[["title", "target"]].apply(lambda x: x[0]+' '+x[1], axis=1).to_list()

#############

# def get_test_tweets():
#     return pd.read_csv(os.path.join(TESTPATH, "tweets-test.tsv"), sep="\t", names=["id","tweet"])

# def get_tweets():
#     return pd.read_csv(os.path.join(DATAPATH, "tweets-train-dev.tsv"), sep="\t", names=["id","tweet"]), get_test_tweets()

# def get_test_qrels():
#     conn_names = ["tweet_id", "tweet_num", "claim_id", "claim_num"]
#     test_conns = pd.read_csv(os.path.join(TESTPATH, "qrels-test.tsv"), sep="\t", names=conn_names)
#     return test_conns

# def get_qrels(include_negatives=True):
#     conn_names = ["tweet_id", "tweet_num", "claim_id", "claim_num"]
#     train_conns = pd.read_csv(os.path.join(DATAPATH, "qrels-train.tsv"), sep="\t", names=conn_names)
#     dev_conns = pd.read_csv(os.path.join(DATAPATH, "qrels-dev.tsv"), sep="\t", names=conn_names)
    
#     # get negatives
#     if include_negatives:
#         top_negative_ranks = np.load("experiments/candidate_selection/shared_resources/train_negative_ranks.npy")
#         train_conns["negative_claim_idx"] = top_negative_ranks

#         dev_top_negative_ranks = np.load("experiments/candidate_selection/shared_resources/dev_negative_ranks.npy")
#         dev_conns["negative_claim_idx"] = dev_top_negative_ranks
    
#     return train_conns, dev_conns, get_test_qrels()

# def get_claims(claimpath=os.path.join(DATAPATH, "vclaims")):
#     claimpaths = [os.path.join(claimpath, f) for f in os.listdir(claimpath)]
#     def load_claim(path):
#         with open(path) as f:
#             return json.load(f)
    
#     claims = [load_claim(path) for path in claimpaths]
#     return pd.DataFrame(claims, columns=["title","subtitle","author","date","vclaim_id","vclaim"])