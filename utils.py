import torch
import torch.nn as nn
import torch.nn.functional as F


def label_to_index(labels, word):
    return torch.tensor(labels.index(word))

def index_to_label(labels, index):
    return labels[index]

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn(batch):
    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, label, _ in batch:
        tensors += [waveform]
        targets += [label]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)