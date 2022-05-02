import torch

def square_mask(mask):
    return torch.einsum("bi,bj->bij", mask, mask)

def ones_mask(tsr, mask_type="flat"):
    batch_size = tsr.shape[0]
    seq_len = tsr.shape[1]
    mask = torch.ones((batch_size, seq_len), 
                      dtype=torch.long, device=tsr.device)
    if mask_type == "flat":
        return mask
    elif mask_type == "square":
        return square_mask(mask)
    else:
        raise ValueError("mask type must be flat or square")

def cat_masks(masks):
    assert len(masks) > 0
    batch_size = masks[0].shape[0]
    seq_lens = [mask.shape[1] for mask in masks]
    total_seq_len = sum(seq_lens)
    total_mask = torch.zeros((batch_size, total_seq_len, total_seq_len), 
                             dtype=torch.long, device=masks[0].device)
    idx = 0
    for seq_len, mask in zip(seq_lens, masks):
        total_mask[:, idx:idx+seq_len, idx:idx+seq_len] = mask
        idx += seq_len
    return total_mask   

    
def multi_tsr_mask(tsrs):
    batch_size = tsrs[0].shape[0]
    masks = [ones_mask(tsr, mask_type="square") for tsr in tsrs]
    return cat_masks(masks)

