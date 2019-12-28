def train_model(n_epochs=4, accumulation_step=2, **kwargs):
    
    optimizer = Adam(model.parameters(), lr=0.001)
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

        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, accumulation_step)
        val_loss, val_score = validation(model, criterion, valid_loader)
    
#         if val_score > best_valid_score:
#             best_valid_score = val_score
#             torch.save(model.state_dict(), 'best_score{}.pt'.format(fold))
    
        elapsed = time.time() - start_time
        
        lr = [_['lr'] for _ in optimizer.param_groups]
        print("Epoch {} - train_loss: {:.6f}  val_loss: {:.6f}  val_score: {:.6f}  lr: {:.5f}  time: {:.0f}s".format(
                epoch+1, train_loss, val_loss, val_score, lr[0], elapsed))

        # inference
        test_preds = inference_test(model, test_loader)
        total_preds.append(test_preds)
        
        # scheduler update
        scheduler.step()
    
    total_preds = np.average(total_preds, weights=checkpoint_weights, axis=0)

    return total_preds, val_score, val_loss