import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

from dynamicquery import utils

class Clef2021Dataset(TensorDataset):
    def __init__(self, encode_fn, claims, tweets, connections, with_negatives=False):
        # claims.vclaim = claims[["title", "subtitle", "vclaim"]].apply(lambda x: f"title: {x[0]}\nsubtitle: {x[1]}\nclaim: {x[2]}", axis=1)
        self.with_negatives = with_negatives
        run_tweets = tweets.join(connections.set_index("tweet_id"), on="id", how="inner")
        run_tweets = run_tweets.join(claims.set_index("vclaim_id"), on="claim_id", how="inner")
        run_tweets = run_tweets[["tweet", "vclaim"]].reset_index()
        run_tweets["encoded_tweet"] = run_tweets.tweet.apply(encode_fn)
        run_tweets["encoded_vclaim"] = run_tweets.vclaim.apply(encode_fn)
        if self.with_negatives:
            # print(claims.iloc[connections.negative_claim_idx].vclaim.apply(encode_fn).to_numpy())
            run_tweets["encoded_nclaim"] = claims.iloc[connections.negative_claim_idx].vclaim.apply(encode_fn).to_numpy()
        self.claim_idx = [claims.vclaim.to_list().index(t_claim) for t_claim in run_tweets.vclaim.to_list()]
        self.data = run_tweets
        
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        X = self.data.encoded_tweet[index]
        X = (np.array(X["input_ids"]), np.array(X["attention_mask"]))
        Y = self.data.encoded_vclaim[index]
        Y = (np.array(Y["input_ids"]), np.array(Y["attention_mask"]))
        if self.with_negatives:
            Y_neg = self.data.encoded_nclaim[index]
            Y = (*Y, np.array(Y_neg["input_ids"]), np.array(Y_neg["attention_mask"]))
        return (X, Y)
    
    
def get_clef2021_dataloader(encode_fn, 
                            claims, 
                            tweets, 
                            connections, 
                            with_negatives=False, 
                            params={'batch_size':32, 'shuffle':True}):
    dataset = Clef2021Dataset(encode_fn, claims, tweets, connections, with_negatives=with_negatives)
    return DataLoader(dataset, **params)
    

    

        