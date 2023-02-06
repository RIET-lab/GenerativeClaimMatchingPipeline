import pandas as pd
import json
import os
import numpy as np
from dynamicquery import defaults

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
    elif dataset == "that-is-a-known-lie-snopes":
        train_queries = get_queries(defaults.KNOWN_LIE_TRAIN_QUERY_PATH)
        train_qrels = get_qrels(defaults.KNOWN_LIE_TRAIN_QREL_PATH)
        targets = pd.read_csv(defaults.KNOWN_LIE_TARGETS_PATH, sep="\t", names=["target_id", "target"]).drop_duplicates()
        
        if test_data:
            test_queries = get_queries(defaults.KNOWN_LIE_TEST_QUERY_PATH)
            test_qrels = get_qrels(defaults.KNOWN_LIE_TEST_QREL_PATH)
            test_data = (test_queries, test_qrels)
        else:
            test_data = None

        dev_queries = train_queries.copy(deep=True)
        dev_qrels = train_qrels.copy(deep=True)

        if negatives_path is not None:
            top_negative_ranks = np.load(os.path.join(negatives_path, "ranks_train_negatives.npy"))
            train_qrels["negative_target_idx"] = top_negative_ranks

            dev_top_negative_ranks = np.load(os.path.join(negatives_path, "ranks_dev_negatives.npy"))
            dev_qrels["negative_target_idx"] = dev_top_negative_ranks
        
        return dict(
            queries = (train_queries, dev_queries),
            qrels = (train_qrels, dev_qrels),
            targets = targets,
            test = test_data
        )
        

    elif dataset == "that-is-a-known-lie-politifact":
        pass
    else:
        raise ValueError(f"{dataset} isnt a default. please include paths manually")

def get_bm25_preprocess_fn(dataset):
    if dataset == "clef2021-checkthat-task2a--english":
        return lambda targets: targets[["title", "subtitle", "target"]].apply(lambda x: x[0]+' '+x[1]+' '+x[2], axis=1).to_list()
    
    elif dataset == "clef2022-checkthat-task2b--english":
        return lambda targets: targets[["title", "target"]].apply(lambda x: x[0]+' '+x[1], axis=1).to_list()

    elif dataset == "clef2022-checkthat-task2a--arabic":
        return lambda targets: targets[["title", "target"]].apply(lambda x: x[0]+' '+x[1], axis=1).to_list()

    elif dataset == "that-is-a-known-lie-snopes":
        return lambda targets: targets.target.to_list()

    elif dataset == "that-is-a-known-lie-politifact":
        pass