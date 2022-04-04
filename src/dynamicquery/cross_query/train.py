import re
import torch

is_adapter = lambda name: re.match("roberta.encoder\.adapter_layer\.", name)
def freeze(module):
    for p in module.parameters(): p.requires_grad = False
def unfreeze(module):
    for p in module.parameters(): p.requires_grad = True

def train(model, 
          optimizer, 
          device,
          train_dataloader,
          dev_dataloader,
          epochs=1,
          print_steps=5,
          adapters_only=False, 
          cls_train=True,
          save_path="./saved_model.pt"):
        
    for name, param in model.named_modules():
        if adapters_only and not is_adapter(name):
            freeze(param)
        else:
            unfreeze(param)
    if cls_train:
        unfreeze(model.classifier)
    else:
        freeze(model.classifier)

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
                "extended_states": external_inputs
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
                "extended_states": external_inputs
            }                

            with torch.no_grad():
                outputs = model(**inpt_dict)
                loss = outputs.loss
                running_loss += loss.item()

        print(f'DEV [{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(dev_dataloader):.3f}')

    print('Finished Training')
    torch.save(model.state_dict(), save_path)