import torch
from transformers import Trainer, TrainingArguments

nn = torch.nn

########################################################################
################## Loss Utils ##########################################
########################################################################

def extract_loss(lm_logits, labels):
    lm_logits = lm_logits.to(torch.float32)

    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.reshape((lm_logits.shape[0], lm_logits.shape[1]-1))
    return loss

def convert_to_negative_loss(loss, labels):
    new_loss = -torch.log(1 - torch.exp(-loss))
    return torch.where(labels==-100, loss, new_loss)

########################################################################
################## Trainers ############################################
########################################################################

class NL3UTrainer(Trainer):
    """Trainer for NL3U loss
    """
    @staticmethod
    def cat_negatives(input):
        return torch.cat([input[:,0], input[:,1]], dim=0)

    def _compute_loss(self, logits, labels):
        batch_size = logits.shape[0] // 2 # accounting for negatives
        loss_vec = extract_loss(logits, labels)
        pos_loss = loss_vec[:batch_size]
        pos_loss = pos_loss.sum() / (labels[:batch_size, 1:] != -100).sum()
        neg_loss = convert_to_negative_loss(loss_vec[batch_size:], labels[batch_size:, 1:])
        neg_loss = neg_loss.sum() / (labels[batch_size:, 1:] != -100).sum()
        loss = pos_loss + neg_loss
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = self.cat_negatives(inputs.get("input_ids"))
        attention_mask = self.cat_negatives(inputs.get("attention_mask"))
        labels = self.cat_negatives(inputs.get("labels"))
        position_ids = self.cat_negatives(inputs.get("position_ids"))

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        outputs = model(**model_inputs)
        # batch_size = outputs.logits.shape[0] // 2 # accounting for negatives
        # loss_vec = extract_loss(outputs.logits, labels)
        # pos_loss = loss_vec[:batch_size]
        # pos_loss = pos_loss.sum() / (labels[:batch_size, 1:] != -100).sum()
        # neg_loss = convert_to_negative_loss(loss_vec[batch_size:], labels[batch_size:, 1:])
        # neg_loss = neg_loss.sum() / (labels[batch_size:, 1:] != -100).sum()
        # loss = pos_loss + neg_loss
        loss = self._compute_loss(outputs.logits, labels)

        return (loss, outputs) if return_outputs else loss

class RLLTrainer(Trainer):
    """Trainer for RLL loss
    """
    def __init__(self, loss_constant, *args, **kwargs):
        self.loss_const = loss_constant
        super().__init__(*args, **kwargs)

    @staticmethod
    def cat_negatives(input):
        return torch.cat([input[:,0], input[:,1]], dim=0)

    def _compute_loss(self, logits, labels):
        batch_size = logits.shape[0] // 2 # accounting for negatives
        
        loss_vec = extract_loss(logits, labels)
        pos_loss = loss_vec[:batch_size]
        neg_loss = loss_vec[batch_size:]

        pos_loss = pos_loss.sum(-1) / (labels[:batch_size, 1:] != -100).sum(-1)
        neg_loss = neg_loss.sum(-1) / (labels[batch_size:, 1:] != -100).sum(-1)

        loss = torch.maximum(
            torch.tensor(0, device=loss_vec.device), 
            pos_loss - neg_loss + self.loss_const
        )
        loss = loss.mean()
        return loss


    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = self.cat_negatives(inputs.get("input_ids"))
        attention_mask = self.cat_negatives(inputs.get("attention_mask"))
        labels = self.cat_negatives(inputs.get("labels"))
        position_ids = self.cat_negatives(inputs.get("position_ids"))

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        outputs = model(**model_inputs)
        # batch_size = outputs.logits.shape[0] // 2 # accounting for negatives
        
        # loss_vec = extract_loss(outputs.logits, labels)
        # pos_loss = loss_vec[:batch_size]
        # neg_loss = loss_vec[batch_size:]

        # pos_loss = pos_loss.sum(-1) / (labels[:batch_size, 1:] != -100).sum(-1)
        # neg_loss = neg_loss.sum(-1) / (labels[batch_size:, 1:] != -100).sum(-1)

        # loss = torch.maximum(
        #     torch.tensor(0, device=loss_vec.device), 
        #     pos_loss - neg_loss + self.loss_const
        # )
        # loss = loss.mean()     
        loss = self._compute_loss(outputs.logits, labels)   

        return (loss, outputs) if return_outputs else loss

class HingedMutualInformationTrainer(Trainer):
    """MI loss hinged
    """
    def __init__(self, max_length, *args, **kwargs):
        self.max_length = max_length
        super().__init__(*args, **kwargs)

    @staticmethod
    def cat_negatives(input):
        return torch.cat([input[:,0], input[:,1]], dim=0)

    def split_losses(self, loss_vec, labels):
        max_length = self.max_length
        loss_left = loss_vec[:, :max_length-1].sum(-1) / (labels[:, 1:max_length] != -100).sum(-1)
        loss_right = loss_vec[:, max_length:].sum(-1) / (labels[:, max_length+1:] != -100).sum(-1)
        # loss = loss_right - loss_left
        return loss_left, loss_right

    def _compute_loss(self, logits, labels):
        loss_vec = extract_loss(logits, labels)
        batch_size = logits.shape[0] // 2

        loss_c, loss_t_c = self.split_losses(loss_vec[:batch_size], labels[:batch_size])
        loss_t, loss_c_t = self.split_losses(loss_vec[batch_size:], labels[batch_size:])

        zero_tensor = torch.tensor(0, dtype=torch.float32, device=loss_vec.device)
        loss_const = 2

        diff_tc = loss_t_c - loss_t
        loss_mi_tc = torch.where(diff_tc + loss_const > 0,
            diff_tc,
            loss_t_c
        )
        diff_ct = loss_c_t - loss_c
        loss_mi_ct = torch.where(diff_ct + loss_const > 0,
            diff_ct,
            zero_tensor
        )
        loss = loss_mi_tc.mean() + loss_mi_ct.mean()
        return loss


    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = self.cat_negatives(inputs.get("input_ids"))
        attention_mask = self.cat_negatives(inputs.get("attention_mask"))
        labels = self.cat_negatives(inputs.get("labels"))
        position_ids = self.cat_negatives(inputs.get("position_ids"))

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        outputs = model(**model_inputs)
        # loss_vec = extract_loss(outputs.logits, labels)
        # batch_size = outputs.logits.shape[0] // 2

        # loss_c, loss_t_c = self.split_losses(loss_vec[:batch_size], labels[:batch_size])
        # loss_t, loss_c_t = self.split_losses(loss_vec[batch_size:], labels[batch_size:])

        # zero_tensor = torch.tensor(0, dtype=torch.float32, device=loss_vec.device)
        # loss_const = 2

        # diff_tc = loss_t_c - loss_t
        # loss_mi_tc = torch.where(diff_tc + loss_const > 0,
        #     diff_tc,
        #     loss_t_c
        # )
        # diff_ct = loss_c_t - loss_c
        # loss_mi_ct = torch.where(diff_ct + loss_const > 0,
        #     diff_ct,
        #     zero_tensor
        # )
        # loss = loss_mi_tc.mean() + loss_mi_ct.mean()
        loss = self._compute_loss(outputs.logits, labels)

        return (loss, outputs) if return_outputs else loss

class MixedTrainer(NL3UTrainer, HingedMutualInformationTrainer):
    task_map = {
        "nl3u": NL3UTrainer,
        "mutual_information": HingedMutualInformationTrainer,
        "rll": RLLTrainer
    }

    @staticmethod
    def cat_negatives(input):
        return torch.cat([input[:,0], input[:,1], input[:,2]], dim=0)

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = self.cat_negatives(inputs.get("input_ids"))
        attention_mask = self.cat_negatives(inputs.get("attention_mask"))
        labels = self.cat_negatives(inputs.get("labels"))
        position_ids = self.cat_negatives(inputs.get("position_ids"))

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        outputs = model(**model_inputs)
        logits = outputs.logits
        batch_size = logits.shape[0] // 3

        l1 = self.task_map.get("mutual_information")._compute_loss(self, 
            logits[:2*batch_size], 
            labels[:2*batch_size])
        l2 = self.task_map.get("nl3u")._compute_loss(self, 
            torch.cat([logits[:batch_size], logits[-batch_size:]], dim=0),
            torch.cat([labels[:batch_size], labels[-batch_size:]], dim=0))
        
        loss = l1 + l2
        return (loss, outputs) if return_outputs else loss

        

##################################################################################
##################################################################################
##################################################################################

class TestTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        print([(k,v.shape) for k,v in inputs.items()])