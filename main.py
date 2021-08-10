import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from tqdm import tqdm
from torchaudio.datasets import SPEECHCOMMANDS
import os
from dataset import SubsetSC, small_dataset
from models import *
from utils import *
import torchvision.models as models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='word')
parser.add_argument('--hard_split', type=bool, default=False)
args = parser.parse_args()

mode = args.mode
hard_split = args.hard_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

dataset = small_dataset('./raw_wave', mode, hard_split)

if hard_split:
    validation_split = 0.111112
else:
    validation_split = 0.2

dataset_size = len(dataset)
shuffle_dataset = True

if not(hard_split):
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(214)
        np.random.shuffle(indices)

train_indices = []
val_indices = []

if hard_split:
    for i, data in enumerate(dataset):
        if (data[1][1] != '10') or not(data[1][0] in ['before', 'close', 'continue', 'fast']):
            train_indices.append(i)
        else:
            val_indices.append(i)
    dataset.remove()

else:
    train_indices, val_indices = indices[split:], indices[:split]

labels = dataset.to_index()
print(labels)
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

batch_size = 16
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                           collate_fn=collate_fn, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                           collate_fn=collate_fn, sampler=valid_sampler)

sample_rate = dataset[0][2]
new_sample_rate = 1000
waveform = dataset[0][0]
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
spec = torchaudio.transforms.Spectrogram(n_fft=64).to(device)
data = next(iter(train_loader))
db = torchaudio.transforms.AmplitudeToDB()
m = db(spec(data[0].cuda())).mean(axis=0)
s = db(spec(data[0].cuda())).std(axis=0)
transformed = torchaudio.transforms.Spectrogram(n_fft=64)(waveform)

def train(model, epoch, log_interval):
    pbar = tqdm(train_loader)
    model.train()
    correct = 0
    losses = []
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device)
        target = target.to(device)

        data = spec(data)
        data = db(data)
        data = (data - m)/s
        output = model(data)
        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print training stats
        if batch_idx % log_interval == 0:
            print(f"\nTrain Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)*(1-validation_split)} ({100. * batch_idx / len(train_loader)*(1-validation_split):.0f}%)]\tLoss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)
        # record loss
        losses.append(loss.item())
    print(f"\nTrain Epoch: {epoch}\tAccuracy: {correct}/{len(train_loader.dataset)*(1-validation_split)} ({100. * correct / (len(train_loader.dataset)*(1-validation_split)):.0f}%)\n")
    # count number of correct predictions
    accuracy = 100. * correct / (len(train_loader.dataset)*(1-validation_split))
    return sum(losses)/len(losses), accuracy

import pdb; pdb.set_trace()
model = FC_Net(n_input=transformed.shape[0], n_output=len(labels), batch_size=batch_size)

model.to(device)
model.train()
n = count_parameters(model)
print("Number of parameters: %s" % n)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


def test(model, epoch, test_loader_):
    model.eval()
    correct = 0
    pbar = tqdm(test_loader_)
    losses = []
    for data, target in pbar:
        data = data.to(device)
        target = target.to(device)        
        data = spec(data)
        data = db(data)
        data = (data - m)/s
        output = model(data)
        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)
        
        loss = F.nll_loss(output.squeeze(), target)
        losses.append(loss.item())
        
        # update progress bar
        pbar.update(pbar_update)

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader_.dataset)*validation_split} ({100. * correct / (len(test_loader_.dataset)*validation_split):.0f}%)\n")
    accuracy = 100. * correct / (len(test_loader_.dataset)*validation_split)
    return  sum(losses)/len(losses), accuracy

log_interval = 300
n_epoch = 100

pbar_update = 1 / (len(train_loader) + len(test_loader))

# The transform needs to live on the same device as the model and the data.
transform = transform.to(device)
max_patient = 20
writer = SummaryWriter()

with tqdm(total=n_epoch) as pbar:
    max_acc = 0
    early_count = 0
    max_epoch = 0
    for epoch in range(1, n_epoch + 1):
        loss_train, accuracy_train = train(model, epoch, log_interval)
        loss_test, accuracy_test = test(model, epoch, test_loader)
        if max_acc < accuracy_test:
            max_acc = accuracy_test
            early_count = 0
            max_epoch = epoch
        else:
            early_count += 1
            if early_count == max_patient:
                break
        writer.add_scalar('Loss/train', loss_train, epoch)
        writer.add_scalar('Loss/test',loss_test, epoch)
        writer.add_scalar('Accuracy/train', accuracy_train, epoch)
        writer.add_scalar('Accuracy/test', accuracy_test, epoch)
        scheduler.step()
print(f'\n Max Test Accuracy is {max_acc} when epoch is {max_epoch}') 
