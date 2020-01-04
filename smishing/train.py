import time
import numpy as np
import torch
from torch import nn, cuda
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score


batch_size = 256


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_model(model, train_loader, valid_loader, criterion, save_path, device, n_epochs=4, learnings_rate=0.001):
    
    optimizer = Adam(model.parameters(), lr=learnings_rate)
    scheduler = LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)
    
    best_epoch = -1
    best_valid_score = 0.
    best_valid_loss = 1.
    all_train_loss = []
    all_valid_loss = []
    
    for epoch in range(n_epochs):
        
        start_time = time.time()

        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device)
        val_loss, val_score = validation(model, criterion, valid_loader, device)
    
        if val_score > best_valid_score:
            best_valid_score = val_score
            torch.save(model.state_dict(), save_path)
    
        elapsed = time.time() - start_time
        
        lr = [_['lr'] for _ in optimizer.param_groups]
        print("Epoch {} - train_loss: {:.6f}  val_loss: {:.6f}  val_score: {:.8f}  lr: {:.5f}  time: {:.0f}s".format(
                epoch+1, train_loss, val_loss, val_score, lr[0], elapsed))

        # scheduler update
        scheduler.step()


def train_one_epoch(model, criterion, train_loader, optimizer, device):
    
    model.train()
    train_loss = 0.
    
    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(train_loader):

        inputs[0] = inputs[0].to(device)
        inputs[1] = inputs[1].to(device)
        targets = targets.to(device)

        preds = model(inputs[0])
        loss = criterion(preds, targets)

        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item() / len(train_loader)
        
    return train_loss


def validation(model, criterion, valid_loader, device):
    
    model.eval()
    valid_preds = np.zeros((len(valid_loader.dataset), 1))
    valid_targets = np.zeros((len(valid_loader.dataset), 1))
    val_loss = 0.
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(valid_loader):
            
            valid_targets[i * batch_size: (i+1) * batch_size] = targets.numpy().copy()

            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
            targets = targets.to(device)
            
            outputs = model(inputs[0])
            loss = criterion(outputs, targets)
            
            valid_preds[i * batch_size: (i+1) * batch_size] = sigmoid(outputs.detach().cpu().numpy())
            
            val_loss += loss.item() / len(valid_loader)
        
    val_score = roc_auc_score(valid_targets, valid_preds)
    
    
    return val_loss, val_score   