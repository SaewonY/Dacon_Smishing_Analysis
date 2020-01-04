import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.utils import shuffle


class SequenceBucketCollator():
    def __init__(self, choose_length, maxlen, sequence_index, length_index, label_index=None):
        self.choose_length = choose_length
        self.sequence_index = sequence_index
        self.length_index = length_index
        self.label_index = label_index
        self.maxlen = maxlen
        
    def __call__(self, batch):
        batch = [torch.stack(x) for x in list(zip(*batch))]
        
        sequences = batch[self.sequence_index]
        lengths = batch[self.length_index]
        
        length = self.choose_length(lengths)
        mask = torch.arange(start=self.maxlen, end=0, step=-1) < length
        padded_sequences = sequences[:, mask]
        
        batch[self.sequence_index] = padded_sequences
        
        if self.label_index is not None:
            return [x for i, x in enumerate(batch) if i != self.label_index], batch[self.label_index]
    
        return batch


def make_loader(x_padded, lengths, y, maxlen=236, batch_size=512, is_train=True):

    dataset = TensorDataset(x_padded, lengths, torch.tensor(y))
    
    collator = SequenceBucketCollator(lambda length: length.max(),
                                        maxlen=maxlen, 
                                        sequence_index=0, 
                                        length_index=1, 
                                        label_index=2)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)                                        
    
    return loader                            
