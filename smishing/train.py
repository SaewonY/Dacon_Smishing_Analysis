import time
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, ReduceLROnPlateau


def train_model(model, train_loader, valid_loader, criterion, model_path, n_epochs=4, use_cuda=False):
    
    optimizer = Adam(model.parameters(), lr=0.005)
    scheduler = LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)
    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
    
    best_epoch = -1
    best_valid_score = 0.
    best_valid_loss = 1.
    all_train_loss = []
    all_valid_loss = []
    total_preds = []
    
    for epoch in range(n_epochs):
        
        start_time = time.time()

        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, use_cuda)
        val_loss, val_score = validation(model, criterion, valid_loader, use_cuda)
    
        if val_score > best_valid_score:
            best_valid_score = val_score
            SAVE_PATH = model_path + 'best_score_fold_{}.pt'.format(fold+1)
            torch.save(model.state_dict(), SAVE_PATH)
    
        elapsed = time.time() - start_time
        
        lr = [_['lr'] for _ in optimizer.param_groups]
        print("Epoch {} - train_loss: {:.6f}  val_loss: {:.6f}  val_score: {:.6f}  lr: {:.5f}  time: {:.0f}s".format(
                epoch+1, train_loss, val_loss, val_score, lr[0], elapsed))

        # scheduler update
        scheduler.step()
    
    total_preds = np.average(total_preds, weights=checkpoint_weights, axis=0)

    return total_preds, val_score


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
    valid_preds = np.zeros((len(valid_dataset), 1))
    valid_targets = np.zeros((len(valid_dataset), 1))
    val_loss = 0.
    
    with torch.no_grad():
        for i, (_, inputs, targets) in enumerate(valid_loader):
            
            targets = targets.unsqueeze(1)

            if use_cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()   
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            valid_preds[i * batch_size: (i+1) * batch_size] = sigmoid(outputs.detach().cpu().numpy())
            valid_targets[i * batch_size: (i+1) * batch_size] = targets.numpy().copy()
            
            val_loss += loss.item() / len(valid_loader)
    
    val_score = roc_auc_score(valid_targets, valid_preds)
    
    
    return val_loss, val_score   