import torch
nn = torch.nn

def mnr_loss(temp=.05):
    ce_loss = nn.CrossEntropyLoss()
    def _mnr_loss(left_tensors, right_tensors):
        logits = torch.einsum("bd,cd->bc", left_tensors, right_tensors)
        return ce_loss(logits/temp, torch.arange(logits.shape[0]).to(logits.device))
    return _mnr_loss
    
    
    