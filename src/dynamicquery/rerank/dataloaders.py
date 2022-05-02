import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

########################################################################
################## Dataset for pretraining #############################
########################################################################

class Clef2021PretrainingDataset(TensorDataset):
    def __init__(self, 
                 encode_fn, 
                 claims,
                 claim_embeddings,
                 n_negatives=5):
        self.n_negatives = n_negatives
        self.claim_embeddings = claim_embeddings
        self.claims = claims
        self.claims["encoded_vclaim"] = self.claims.vclaim.apply(encode_fn)
        
    def __len__(self):
        return len(self.claims)
    
    def __getitem__(self, index):
        neg_idx_set = set(range(len(self.claim_embeddings)))
        neg_idx_set.discard(index)
        negative_idx = np.random.choice(np.array(list(neg_idx_set)), size=(self.n_negatives,), replace=False)
        # negative_idx = np.random.randint(len(self.claim_embeddings), size=(self.n_negatives,))
        negative_embs = self.claim_embeddings[negative_idx]
        positive_emb = self.claim_embeddings[index]
        embs = np.concatenate([np.expand_dims(positive_emb, 0), negative_embs], 0)

        positive_tokens = self.claims["encoded_vclaim"][index]
        positive_inpt = (np.array(positive_tokens["input_ids"]), np.array(positive_tokens["attention_mask"]))

        return (positive_inpt, embs)
    
    
def get_clef2021_pretraining_dataloader(encode_fn, 
                            claims, 
                            claim_embeddings,
                            n_negatives=5,
                            params={'batch_size':32, 'shuffle':True}):
    dataset = Clef2021PretrainingDataset(encode_fn, 
                              claims, 
                              claim_embeddings,
                              n_negatives=n_negatives)
    # return dataset
    return DataLoader(dataset, **params)

########################################################################
################## Dataset for training ################################
########################################################################

class Clef2021RerankedDataset(TensorDataset):
    def __init__(self, 
                 encode_fn, 
                 claims, 
                 tweets, 
                 connections,
                 claim_embeddings,
                 ranks,
                 tweet_embeddings=None):
        self.claim_embeddings = claim_embeddings
        self.tweet_embeddings = tweet_embeddings
        self.ranks = ranks
        run_tweets = tweets.join(connections.set_index("tweet_id"), on="id", how="inner")
        run_tweets = run_tweets.join(claims.set_index("vclaim_id"), on="claim_id", how="inner")
        run_tweets = run_tweets[["tweet", "vclaim"]].reset_index()
        run_tweets["encoded_tweet"] = run_tweets.tweet.apply(encode_fn)
        self.claim_idx = [claims.vclaim.to_list().index(t_claim) for t_claim in run_tweets.vclaim.to_list()]
        self.data = run_tweets
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        Xt = self.data.encoded_tweet[index]
        Xt = (np.array(Xt["input_ids"]), np.array(Xt["attention_mask"]))
        Xe = self.claim_embeddings[self.ranks[index]]
        Ye = self.claim_embeddings[self.claim_idx[index:index+1]]
        if not self.tweet_embeddings is None:
            Ye_tweet = self.tweet_embeddings[index:index+1]
            return (Xt, np.concatenate([Ye, Xe, Ye_tweet], axis=0))
        else:
            return (Xt, np.concatenate([Ye, Xe], axis=0))
    
    
def get_clef2021_reranked_dataloader(encode_fn, 
                            claims, 
                            tweets, 
                            connections, 
                            claim_embeddings,
                            ranks,
                            tweet_embeddings=None,
                            params={'batch_size':32, 'shuffle':True}):
    dataset = Clef2021RerankedDataset(encode_fn, 
                              claims, 
                              tweets, 
                              connections, 
                              claim_embeddings,
                              ranks,
                              tweet_embeddings)
    return DataLoader(dataset, **params)

########################################################################
################## Dataset for eval ####################################
########################################################################

class Clef2021RerankedEvalDataset(TensorDataset):
    def __init__(self, 
                 encode_fn, 
                 claims, 
                 tweets, 
                 connections,
                 claim_embeddings,
                 ranks,
                 tweet_embeddings=None):
        self.claim_embeddings = claim_embeddings
        self.tweet_embeddings = tweet_embeddings
        self.ranks = ranks
        run_tweets = tweets.join(connections.set_index("tweet_id"), on="id", how="inner")
        run_tweets = run_tweets.join(claims.set_index("vclaim_id"), on="claim_id", how="inner")
        run_tweets = run_tweets[["tweet", "vclaim"]].reset_index()
        run_tweets["encoded_tweet"] = run_tweets.tweet.apply(encode_fn)
        self.claim_idx = [claims.vclaim.to_list().index(t_claim) for t_claim in run_tweets.vclaim.to_list()]
        self.data = run_tweets
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        Xt = self.data.encoded_tweet[index]
        Xt = (np.array(Xt["input_ids"]), np.array(Xt["attention_mask"]))
        Xe = self.claim_embeddings[self.ranks[index]]
        if not self.tweet_embeddings is None:
            Ye_tweet = self.tweet_embeddings[index:index+1]
            return (Xt, np.concatenate([Xe, Ye_tweet], axis=0))
        else:
            return (Xt, Xe)
    
    
def get_clef2021_reranked_eval_dataloader(encode_fn, 
                            claims, 
                            tweets, 
                            connections, 
                            claim_embeddings,
                            ranks,
                            tweet_embeddings=None,
                            params={'batch_size':32, 'shuffle':True}):
    dataset = Clef2021RerankedEvalDataset(encode_fn, 
                              claims, 
                              tweets, 
                              connections, 
                              claim_embeddings,
                              ranks,
                              tweet_embeddings=tweet_embeddings)
    return DataLoader(dataset, **params)  


########################################################################
################## Dataset for Retro ###################################
########################################################################

class Clef2021RetroDataset(TensorDataset):
    def __init__(self, 
                 encode_fn, 
                 claims, 
                 tweets, 
                 connections,
                 claim_embeddings,
                 ranks):
        self.claim_embeddings = claim_embeddings
        self.ranks = ranks
        run_tweets = tweets.join(connections.set_index("tweet_id"), on="id", how="inner")
        run_tweets = run_tweets.join(claims.set_index("vclaim_id"), on="claim_id", how="inner")
        run_tweets = run_tweets[["tweet", "vclaim"]].reset_index()
        run_tweets["encoded_tweet"] = run_tweets.tweet.apply(encode_fn)
        self.claim_idx = [claims.vclaim.to_list().index(t_claim) for t_claim in run_tweets.vclaim.to_list()]
        self.data = run_tweets
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        Xt = self.data.encoded_tweet[index]
        Xt = (np.array(Xt["input_ids"]), np.array(Xt["attention_mask"]))
        Xe = self.claim_embeddings[self.ranks[index]]
        Ye = self.claim_embeddings[self.claim_idx[index:index+1]]
        return (Xt, np.concatenate([Ye, Xe], axis=0))
    
    
def get_clef2021_retro_dataloader(encode_fn, 
                            claims, 
                            tweets, 
                            connections, 
                            claim_embeddings,
                            ranks,
                            tweet_embeddings=None,
                            params={'batch_size':32, 'shuffle':True}):
    dataset = Clef2021RetroDataset(encode_fn, 
                              claims, 
                              tweets, 
                              connections, 
                              claim_embeddings,
                              ranks)
    return DataLoader(dataset, **params)