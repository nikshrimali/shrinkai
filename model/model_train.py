from tqdm import tqdm
from torch import nn
import torch.nn
from torch.functional import F
import os

# os.chdir('d:\Python Projects\EVA')
# cwd = os.getcwd()

# model_dir = os.path.join(cwd, 'Assignment-6/saved_models/model.pth')

def model_training(model, device, train_dataloader, optimizer, train_acc, train_losses, l1_loss=False):
            
    model.train()
    pbar = tqdm(train_dataloader)
    correct = 0
    processed = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.nll_loss(y_pred, target)

        # IF L1 Loss
        if l1_loss:
            lambda_l1 = 0.0001
            l1 = 0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
                loss = loss + lambda_l1*l1
        
        

        train_losses.append(loss)
        loss.backward()
        optimizer.step()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        # print statistics
        running_loss += loss.item()
        pbar.set_description(desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)
