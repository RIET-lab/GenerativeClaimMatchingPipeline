import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

from dynamicquery import utils

class EncoderDataset(TensorDataset):
    def __init__(self, encode_fn, targets, queries, qrels, with_negatives=False):
        # claims.vclaim = claims[["title", "subtitle", "vclaim"]].apply(lambda x: f"title: {x[0]}\nsubtitle: {x[1]}\nclaim: {x[2]}", axis=1)
        self.with_negatives = with_negatives
        run_queries = queries.merge(qrels, left_on="query_id", right_on="query_id", how="inner")
        run_queries = run_queries.merge(targets, left_on="target_id", right_on="target_id", how="inner")
        run_queries = run_queries[["query", "target"]].reset_index()
        run_queries["encoded_query"] = run_queries['query'].apply(encode_fn)
        run_queries["encoded_target"] = run_queries.target.apply(encode_fn)
        if self.with_negatives:
            run_queries["encoded_negative"] = targets.iloc[qrels.negative_target_idx].target.apply(encode_fn).to_numpy()
        # self.claim_idx = [targets.target.to_list().index(t_claim) for t_claim in run_queries.target.to_list()]
        self.data = run_queries
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        X = self.data.encoded_query[index]
        X = (np.array(X["input_ids"]), np.array(X["attention_mask"]))
        Y = self.data.encoded_target[index]
        Y = (np.array(Y["input_ids"]), np.array(Y["attention_mask"]))
        if self.with_negatives:
            Y_neg = self.data.encoded_negative[index]
            Y = (*Y, np.array(Y_neg["input_ids"]), np.array(Y_neg["attention_mask"]))
        return (X, Y)
    
    
def get_encoder_dataloader(encode_fn, 
                            targets, 
                            queries, 
                            qrels, 
                            with_negatives=False, 
                            params={'batch_size':32, 'shuffle':True}):
    dataset = EncoderDataset(encode_fn, targets, queries, qrels, with_negatives=with_negatives)
    return DataLoader(dataset, **params)
    

    

        