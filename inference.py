def inference_test(model, test_loader):
    model.eval()

    test_preds = np.zeros((len(test_dataset), 1))

    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            if use_cuda:
                inputs[0] = inputs[0].cuda()
                inputs[1] = inputs[1].cuda()
#                 inputs[2] = inputs[2].cuda()

            outputs = model(inputs[0], inputs[1])
#             outputs = model(inputs[0], inputs[1], inputs[2])
            test_preds[i * batch_size:(i+1) * batch_size] = sigmoid(outputs.cpu().numpy())
    
    return test_preds