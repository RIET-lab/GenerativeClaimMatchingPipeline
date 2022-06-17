import os
import pandas as pd
from functools import partial
import numpy as np
import argparse
import configparser
from tqdm import tqdm
import json

from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from dynamicquery import utils
import dataloaders

def get_most_recent_checkpoint_filename(path):
    max_step_number = max([int(ckpt.split("-")[-1]) for ckpt in os.listdir(path)])
    return f"checkpoint-{max_step_number}"

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, 
                        help="path where config.ini lies")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--bm25', action='store_true')
    parser.add_argument('--n-candidates', type=int, default=None,
                        help="overwrite number of candidates")
    parser.add_argument('--checkpoint', type=int, default=None,
                        help="specify ckpt instead of most recent")
    parser.add_argument('--custom', type=str, default=None)
    parser.add_argument('--claims-path', type=str, default=None)
    parser.add_argument('--output', type=str, default='test_submission_0.tsv')


    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))

    # Load Model
    model_str = config["model"].get("model_string")
    ckpt_str = os.path.join(args.experiment_path, config["training"].get("save_dir"))
    ckpt_filename = get_most_recent_checkpoint_filename(ckpt_str)
    if args.checkpoint:
        ckpt_filename = f"checkpoint-{args.checkpoint}"
    ckpt_str = os.path.join(ckpt_str, ckpt_filename)
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

    tweets, test_tweets = utils.get_tweets()
    test_tweets = test_tweets[1:]
    train_conns, dev_conns, test_conns = utils.get_qrels()
    if args.claims_path is None:
        claims = utils.get_claims()
    else:
        print(f"getting claims from {args.claims_path}")
        def get_claims(claimpath):
            claimpaths = [os.path.join(claimpath, f) for f in os.listdir(claimpath)]
            def load_claim(path):
                with open(path) as f:
                    return json.load(f)
            
            claims = [load_claim(path) for path in claimpaths]
            return pd.DataFrame(claims, columns=["title","subtitle","author","date","vclaim_id","vclaim"])

        claims = get_claims(args.claims_path)

    cs_path = config["eval"].get("candidate_selection")
    if args.bm25:
        cs_path = "experiments/candidate_selection/shared_resources"

    if args.train:
        ranks_path = os.path.join(cs_path, "ranks_train.npy")
        ranks = np.load(ranks_path)
        tweets = tweets.join(train_conns.set_index("tweet_id"), on="id", how="inner")
        tweets = tweets.reset_index()

    elif args.dev:
        ranks_path = os.path.join(cs_path, "ranks_dev.npy")
        ranks = np.load(ranks_path)
        tweets = tweets.join(dev_conns.set_index("tweet_id"), on="id", how="inner")
        tweets = tweets.reset_index()

    elif args.test:
        ranks_path = os.path.join(cs_path, "ranks_test.npy")
        ranks = np.load(ranks_path)
        tweets = test_tweets.reset_index()

    elif args.custom is not None:
        ranks_path = os.path.join(cs_path, "custom_queries", "ranks_custom.npy")
        ranks = np.load(ranks_path)
        tweets = pd.read_csv(args.custom, sep="\t", names=["id","tweet"])

    def run_and_save(tweets, ranks):
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
        tweet_tokenize = partial(_tokenize, tokenizer=tokenizer, max_length=max_length, prefix="tweet:")
        claim_tokenize = partial(_tokenize, tokenizer=tokenizer, max_length=max_length, prefix="claim:")

        dataset = dataloaders.ExtendedAutoRegressiveScoringDataset(
            tweet_encode_fn=tweet_tokenize, 
            claim_encode_fn=claim_tokenize,
            claims=claims, 
            tweets=tweets, 
            ranks=ranks,
            n_candidates=N_CANDIDATES
        )

        # Run
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        model.eval()
        batch_size = config["eval"].getint("batch_size", 1)
        def get_reranks(dataset):
            reranks = []
            rescores = []
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
                        loss_vals = loss.cpu().numpy()
                        reranks.extend(loss_vals.argsort().tolist())
                        rescores.extend((-loss_vals).tolist())

                        labels, input_ids, attention_mask, position_ids = [], [], [], []

                    else:
                        continue
            return np.array(reranks), np.array(rescores)

        # train ptn
        return get_reranks(dataset)
    
    reranks, rescores = run_and_save(tweets, ranks)

    def reindex(ranks, indeces):
        partial_ranks = [ranks[i, :N_CANDIDATES][indeces[i]] for i in range(len(indeces))]
        ranks[:, :N_CANDIDATES] = np.stack(partial_ranks, 0)
        return ranks

    ranks = reindex(ranks, reranks)
    scores = reindex(rescores, reranks)
    
    np.save(os.path.join(args.experiment_path, "submission_ranks_test.npy"), ranks)

    submission_rows = list()
    for i, (ranklist, scorelist) in enumerate(zip(ranks, scores)):
        for j, (rank, score) in enumerate(zip(ranklist[:N_CANDIDATES], scorelist[:N_CANDIDATES])):
            row = [tweets.id[i], "Q0", claims.vclaim_id[rank], j+1, score, "riet"]
            submission_rows.append(row)

    submission_df = pd.DataFrame(submission_rows, columns=["tweet_id", "Q0", "vclaim_id", "rank", "score", "tag"])
    submission_df.to_csv(args.output, sep="\t", index=False, header=False)

    
if __name__ == '__main__':
    run()
    
    
    