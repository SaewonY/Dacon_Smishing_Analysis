import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from sklearn.utils import shuffle


class TextDataset(Dataset):

    def __init__(self, seqs, targets=None, maxlen=200):
        if targets is not None:
            self.targets = targets
        else:
            self.targets = np.random.randint(2, size=(len(seqs),))
        
        self.seqs = seqs
        self.maxlen = maxlen
        
    def __len__(self):
        return len(self.seqs)
        
    def get_keys(self):
        lens = np.fromiter(
            ((min(self.maxlen, len(seq))) for seq in self.seqs),
            dtype=np.int32)
        return lens
        
    def __getitem__(self, index):
        return index, self.seqs[index], self.targets[index]


def collate_fn(data):

    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        max_len = max(lens)

        padded_seqs = torch.zeros(len(seqs), max_len).long()
        for i, seq in enumerate(seqs):
            start = max_len - lens[i]
            padded_seqs[i, start:] = torch.LongTensor(seq)
        return padded_seqs

    index, seqs, targets = zip(*data)

    seqs = _pad_sequences(seqs)
    return index, seqs, torch.FloatTensor(targets)


class BucketSampler(Sampler):

    def __init__(self, data_source, sort_keys, bucket_size=None, batch_size=1048, shuffle_data=True):
        super().__init__(data_source)
        self.shuffle = shuffle_data
        self.batch_size = batch_size
        self.sort_keys = sort_keys
        self.bucket_size = bucket_size if bucket_size is not None else len(sort_keys)
        self.weights = None

        if not shuffle_data:
            self.index = self.prepare_buckets()
        else:
            self.index = None

    def set_weights(self, weights):
        assert weights >= 0
        total = np.sum(weights)
        if total != 1:
            weights = weights / total
        self.weights = weights

    def __iter__(self):
        indices = None
        if self.weights is not None:
            total = len(self.sort_keys)
            indices = np.random.choice(total, (total,), p=self.weights)
        if self.shuffle:
            self.index = self.prepare_buckets(indices)
        return iter(self.index)

    def get_reverse_indexes(self):
        indexes = np.zeros((len(self.index),), dtype=np.int32)
        for i, j in enumerate(self.index):
            indexes[j] = i
        return indexes

    def __len__(self):
        return len(self.sort_keys)
        
    def prepare_buckets(self, indices=None):
        lens = - self.sort_keys
        assert self.bucket_size % self.batch_size == 0 or self.bucket_size == len(lens)

        if indices is None:
            if self.shuffle:
                indices = shuffle(np.arange(len(lens), dtype=np.int32))
                lens = lens[indices]
            else:
                indices = np.arange(len(lens), dtype=np.int32)

        #  bucket iterator
        def divide_chunks(l, n):
            if n == len(l):
                yield np.arange(len(l), dtype=np.int32), l
            else:
                # looping till length l
                for i in range(0, len(l), n):
                    data = l[i:i + n]
                    yield np.arange(i, i + len(data), dtype=np.int32), data
    
        new_indices = []
        extra_batch = None
        for chunk_index, chunk in divide_chunks(lens, self.bucket_size):
            # sort indices in bucket by descending order of length
            indices_sorted = chunk_index[np.argsort(chunk, axis=-1)]
            batches = []
            for _, batch in divide_chunks(indices_sorted, self.batch_size):
                if len(batch) == self.batch_size:
                    batches.append(batch.tolist())
                else:
                    assert extra_batch is None
                    assert batch is not None
                    extra_batch = batch
    
            # shuffling batches within buckets
            if self.shuffle:
                batches = shuffle(batches)
            for batch in batches:
                new_indices.extend(batch)
    
        if extra_batch is not None:
            new_indices.extend(extra_batch)
        return indices[new_indices]


def make_loader(X, Y, batch_size=512, max_len=220):

    dataset = TextDataset(X, torch.tensor(Y), maxlen=max_len)
    sampler = BucketSampler(dataset, dataset.get_keys(), bucket_size=batch_size * 20, batch_size=batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=0, collate_fn=collate_fn)

    return loader

