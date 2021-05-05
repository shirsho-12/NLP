from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import torch


def get_vocab(dset):
    counter = Counter()
    tokenizer = get_tokenizer('basic_english')
    for (label, text) in dset:
        counter.update(tokenizer(text))
    return Vocab(counter, min_freq=1), tokenizer


def collate_fn(data, device):
    labels, x_arr, offsets = [], [], []
    vocab, tokenizer = get_vocab(data)
    for (y, x) in data:
        labels.append([(var - 1) for var in y])
        x_processed = torch.tensor([vocab[token] for token in tokenizer(x)], dtype=torch.int64)
        x_arr.append(x_processed)
        offsets.append(x_processed.size(0))
    labels = torch.tensor(labels, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    x_arr = torch.cat(x_arr)
    return labels.to(device), x_arr.to(device), offsets.to(device)
