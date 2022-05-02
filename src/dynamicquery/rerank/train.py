import re
import torch
import numpy as np

def is_adapter(name): 
    check_v1 = re.match("roberta.encoder\.adapter_layer", name)
    check_v2 = name == "roberta.adapter_layer"
    check_v3 = re.match("roberta.encoder\.ca_layer", name)
    return check_v1 or check_v2 or check_v3

def freeze(module):
    for p in module.parameters(): p.requires_grad = False
    
def unfreeze(module):
    for p in module.parameters(): p.requires_grad = True
    
def set_train_adapters(model, train):
    for layer in model.roberta.encoder.ca_layer + [model.roberta.encoder.adapter_layer]:
        if train:
            unfreeze(layer)
        else:
            freeze(layer)
            
def set_train_base(model, train):
    for layer in model.roberta.encoder.layer + [model.roberta.embeddings]:
        if train:
            unfreeze(layer)
        else:
            freeze(layer)

def train(model, 
          optimizer, 
          device,
          train_dataloader,
          dev_dataloader=None,
          epochs=1,
          print_steps=5,
          adapters_only=False, 
          cls_train=True,
          includes_tweet_state=False,
          save_path="./saved_model.pt"):
      
    # for name, param in model.named_modules():
    #     if adapters_only and not is_adapter(name):
    #         freeze(param)
    #     else:
    #         unfreeze(param)
    # set_train_adapters(model, True)
    set_train_base(model, not adapters_only)
    
    if cls_train:
        unfreeze(model.classifier)
    else:
        freeze(model.classifier)
        
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    print("There are {} trainable parameters!".format(n_trainable_params))

    model.to(device)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, external_inputs = data
            inputs = [elmt.to(device) for elmt in inputs]
            external_inputs = external_inputs.to(device)
            
            inpt_dict = {
                "input_ids": inputs[0],
                "attention_mask": inputs[1],
                "extended_states": external_inputs,
                "includes_tweet_state": includes_tweet_state
            }
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(**inpt_dict)
            # embeddings = outputs['last_hidden_state']
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % print_steps  == print_steps-1:    # print every 2000 mini-batches
                print(f'TRAIN [{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_steps:.3f}')
                running_loss = 0.0

        if dev_dataloader:
            running_loss = 0.0   
            model.eval()
            for i, data in enumerate(dev_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, external_inputs = data
                inputs = [elmt.to(device) for elmt in inputs]
                external_inputs = external_inputs.to(device)

                inpt_dict = {
                    "input_ids": inputs[0],
                    "attention_mask": inputs[1],
                    "extended_states": external_inputs,
                    "includes_tweet_state": includes_tweet_state
                }                

                with torch.no_grad():
                    outputs = model(**inpt_dict)
                    loss = outputs.loss
                    running_loss += loss.item()

            print(f'DEV [{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(dev_dataloader):.3f}')

    print('Finished Training Session')
    if save_path:
        torch.save(model.state_dict(), save_path)