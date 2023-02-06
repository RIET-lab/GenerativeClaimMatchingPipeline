import os
import pandas as pd
from functools import partial
import numpy as np
import argparse
import configparser
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dynamicquery import utils, defaults
import dataloaders

def get_most_recent_checkpoint_filename(path):
    max_step_number = max([int(ckpt.split("-")[-1]) for ckpt in os.listdir(path)])
    return f"checkpoint-{max_step_number}"

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, 
                        help="path where config.ini lies")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no-train', action='store_true')
    parser.add_argument('--no-dev', action='store_true')
    parser.add_argument('--bm25', action='store_true')
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--n-candidates', type=int, default=None,
                        help="overwrite number of candidates")
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--candidate-selection', type=str, default=None)
    parser.add_argument('--checkpoint', type=int, default=None,
                        help="specify ckpt instead of most recent")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

    # Load Model
    model_str = config["model"].get("model_string")
    ckpt_str = os.path.join(args.experiment_path, config["training"].get("save_dir"))
    ckpt_filename = get_most_recent_checkpoint_filename(ckpt_str)
    if args.checkpoint:
        ckpt_filename = f"checkpoint-{args.checkpoint}"
    
    if not args.raw:
        ckpt_str = os.path.join(ckpt_str, ckpt_filename)
    else:
        ckpt_str = model_str
    print(f"Loading model from {ckpt_str}")

    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(ckpt_str)

    # Setup Data
    if args.n_candidates is None:
        N_CANDIDATES = config["eval"].getint("n_candidates", 5)
    else: 
        N_CANDIDATES = args.n_candidates
    print(f"Reranking top {N_CANDIDATES}")

    if args.dataset is None:
        dataset = config["data"].get("dataset")
    else:
        dataset = args.dataset
    data = utils.load_data(dataset, test_data=args.test)
    train_queries, dev_queries = data["queries"]
    train_qrels, dev_qrels = data["qrels"]
    targets = data["targets"]

    if args.candidate_selection is None:
        cs_path = config["eval"].get("candidate_selection")
    else:
        cs_path = args.candidate_selection
    if args.bm25:
        cs_path = f"experiments/candidate_selection/bm25/{dataset}"

    train_ranks_path = os.path.join(cs_path, "ranks_train.npy")
    dev_ranks_path = os.path.join(cs_path, "ranks_dev.npy")

    train_ranks = np.load(train_ranks_path)
    dev_ranks = np.load(dev_ranks_path)
    
    if args.test:
        test_ranks_path = os.path.join(cs_path, "ranks_test.npy")
        test_ranks = np.load(test_ranks_path)
        test_queries, test_qrels = data["test"]

    # setup tokenize fn
    tokenizer.pad_token = tokenizer.eos_token
    def _tokenize(text, tokenizer, max_length, prefix=""):
        text = tokenizer.bos_token + prefix + text + tokenizer.eos_token
        token_params = dict(
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_attention_mask=True
        )
        return tokenizer(text, **token_params)

    max_length=config["training"].getint("max_length")
    if dataset == "clef2021-checkthat-task2a--english":
        query_prefix = defaults.TASK_2A_EN_QUERY_PREFIX
        target_prefix = defaults.TASK_2A_EN_TARGET_PREFIX
    elif dataset == "clef2022-checkthat-task2b--english":
        query_prefix = defaults.TASK_2B_EN_QUERY_PREFIX
        target_prefix = defaults.TASK_2B_EN_TARGET_PREFIX
    elif dataset == "clef2022-checkthat-task2a--arabic":
        query_prefix = defaults.TASK_2A_AR_QUERY_PREFIX
        target_prefix = defaults.TASK_2A_AR_TARGET_PREFIX
    elif dataset == "that-is-a-known-lie-snopes":
        query_prefix = defaults.KNOWN_LIE_QUERY_PREFIX
        target_prefix = defaults.KNOWN_LIE_TARGET_PREFIX
    else:
        raise ValueError(f"Dataset {dataset} not implemented yet")

    tweet_tokenize = partial(_tokenize, tokenizer=tokenizer, max_length=max_length, prefix=query_prefix)
    claim_tokenize = partial(_tokenize, tokenizer=tokenizer, max_length=max_length, prefix=target_prefix)

    train_dataset = dataloaders.ExtendedAutoRegressiveEvalDataset(
        tweet_encode_fn=tweet_tokenize, 
        claim_encode_fn=claim_tokenize,
        claims=targets, 
        tweets=train_queries, 
        connections=train_qrels, 
        ranks=train_ranks,
        n_candidates=N_CANDIDATES
    )

    dev_dataset = dataloaders.ExtendedAutoRegressiveEvalDataset(
        tweet_encode_fn=tweet_tokenize, 
        claim_encode_fn=claim_tokenize,
        claims=targets, 
        tweets=dev_queries, 
        connections=dev_qrels, 
        ranks=dev_ranks,
        n_candidates = N_CANDIDATES
    )

    # Run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    batch_size = config["eval"].getint("batch_size", 1)
    def get_reranks(dataset):
        reranks = []
        with torch.no_grad():
            labels, input_ids, attention_mask, position_ids = [], [], [], []
            for i in tqdm(range(len(dataset))):
                item = dataset[i]

                labels.append(torch.tensor(np.tile(item["labels"][np.newaxis], (N_CANDIDATES,1)), device=device))
                input_ids.append(torch.tensor(np.stack(item["input_ids"], 0), device=device))
                attention_mask.append(torch.tensor(np.stack(item["attention_mask"], 0), device=device))
                position_ids.append(torch.tensor(np.tile(item["position_ids"][np.newaxis], (N_CANDIDATES,1)), device=device))

                if i == len(dataset) - 1 or not (i+1) % batch_size:
                    labels = torch.cat(labels, 0)
                    input_ids = torch.cat(input_ids, 0)
                    attention_mask = torch.cat(attention_mask, 0)
                    position_ids = torch.cat(position_ids, 0)

                    inpt_dict = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                        "position_ids": position_ids
                    }
                    outpt = model(**inpt_dict)

                    lm_logits = outpt.logits.to(torch.float32)
                    # Shift so that tokens < n predict n
                    shift_logits = lm_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    loss = loss.reshape((-1, N_CANDIDATES, input_ids.shape[1]-1)).sum(axis=-1)
                    reranks.extend(loss.cpu().numpy().argsort().tolist())

                    labels, input_ids, attention_mask, position_ids = [], [], [], []

                else:
                    continue
        return reranks

    # train ptn
    if not args.no_train:
        reranks = get_reranks(train_dataset)
    if not args.no_dev:
        dev_reranks = get_reranks(dev_dataset)

    # test ptn
    if args.test:
        test_dataset = dataloaders.ExtendedAutoRegressiveEvalDataset(
            tweet_encode_fn=tweet_tokenize, 
            claim_encode_fn=claim_tokenize,
            claims=targets, 
            tweets=test_queries, 
            connections=test_qrels, 
            ranks=test_ranks,
            n_candidates=N_CANDIDATES
        )
        test_reranks = get_reranks(test_dataset)
        
        
    
    def get_idx(connections, claims, tweets):
        run_tweets = tweets.merge(connections, on="query_id", how="inner")
        run_tweets = run_tweets.merge(claims, on="target_id", how="inner")
        run_tweets = run_tweets[["query", "target"]].reset_index()
        claim_idx = [claims.target.to_list().index(t_claim) for t_claim in run_tweets.target.to_list()]
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
    partitions = []
    if not args.no_train: partitions.append("train")
    if not args.no_dev: partitions.append("dev")
    if args.test: partitions.append("test")
    k_values = list(filter(lambda k: k <= N_CANDIDATES, [1,5,10,20]))
    for ptn in partitions:
        if ptn == "train":
            run_tweets, claim_idx = get_idx(train_qrels, targets, train_queries)
            ranks = np.array([ids[rerank] for ids, rerank in zip(train_ranks, reranks)])
        elif ptn == "dev":
            run_tweets, claim_idx = get_idx(dev_qrels, targets, dev_queries)
            ranks = np.array([ids[rerank] for ids, rerank in zip(dev_ranks, dev_reranks)])
        elif ptn == "test":
            run_tweets, claim_idx = get_idx(test_qrels, targets, test_queries)
            ranks = np.array([ids[rerank] for ids, rerank in zip(test_ranks, test_reranks)])

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
    
    
    