
import argparse
import pandas as pd


def inference_test(model, test_loader):
    model.eval()

    test_preds = np.zeros((len(test_dataset), 1))

    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            if use_cuda:
                inputs = inputs.cuda()

            outputs = model(inputs)
            test_preds[i * batch_size:(i+1) * batch_size] = sigmoid(outputs.cpu().numpy())
    
    return test_preds



def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument




if __name__ == '__main__':
    main()