import time
import numpy as np
import torch
from torch import nn, cuda
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

batch_size = 512

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_model(model, train_loader, valid_loader, criterion, save_path, n_epochs=4, use_cuda=False):
    
    optimizer = Adam(model.parameters(), lr=0.005)
    scheduler = LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)
    
    # scale_fn = combine_scale_functions(
    #     [partial(scale_cos, 1e-4, 5e-3), partial(scale_cos, 5e-3, 1e-3)], [0.2, 0.8])
    # scheduler = ParamScheduler(optimizer, scale_fn, n_epochs * len(train_loader))


    best_epoch = -1
    best_valid_score = 0.
    best_valid_loss = 1.
    all_train_loss = []
    all_valid_loss = []
    
    for epoch in range(n_epochs):
        
        start_time = time.time()

        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, use_cuda)
        val_loss, val_score = validation(model, criterion, valid_loader, use_cuda)
    
        if val_score > best_valid_score:
            best_valid_score = val_score
            torch.save(model.state_dict(), save_path)
    
        elapsed = time.time() - start_time
        
        lr = [_['lr'] for _ in optimizer.param_groups]
        print("Epoch {} - train_loss: {:.6f}  val_loss: {:.6f}  val_score: {:.6f}  lr: {:.5f}  time: {:.0f}s".format(
                epoch+1, train_loss, val_loss, val_score, lr[0], elapsed))

        # scheduler update
        scheduler.step()


def train_one_epoch(model, criterion, train_loader, optimizer, use_cuda=False):
    
    model.train()
    train_loss = 0.
    
    optimizer.zero_grad()

    for i, (_, inputs, targets) in enumerate(train_loader):

        targets = targets.unsqueeze(1)

        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()    
            
        preds = model(inputs)
        loss = criterion(preds, targets)

        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item() / len(train_loader)
        
    return train_loss


def validation(model, criterion, valid_loader, use_cuda=False):
    
    model.eval()
    valid_preds = np.zeros((len(valid_loader.dataset), 1))
    valid_targets = np.zeros((len(valid_loader.dataset), 1))
    val_loss = 0.
    
    with torch.no_grad():
        for i, (_, inputs, targets) in enumerate(valid_loader):
            
            targets = targets.unsqueeze(1)
            valid_targets[i * batch_size: (i+1) * batch_size] = targets.numpy().copy()

            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()   
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            valid_preds[i * batch_size: (i+1) * batch_size] = sigmoid(outputs.detach().cpu().numpy())
            
            val_loss += loss.item() / len(valid_loader)
    
    val_score = roc_auc_score(valid_targets, valid_preds)
    
    
    return val_loss, val_score   