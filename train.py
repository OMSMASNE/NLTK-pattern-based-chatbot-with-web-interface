# Copyright (c) 2021 OM SANTOSHKUMAR MASNE.
# All Rights Reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for license information.

import torch
from torch import mode, optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataset

import numpy as np
import json

from utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as file:
    intents = json.load(file)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        word = tokenize(pattern)
        all_words.extend(word)
        xy.append((word, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(word) for word in all_words if word not in ignore_words]

all_words = sorted(set(all_words))
tags = sorted(set(tags))

print("patterns: ", len(xy))
print("Number of tags: ", len(tags))
print("Tags: ", tags)
print("Number of Unique stemmed words: ", len(all_words))
print("Unique stemmed words: ", all_words)

X_train = []
Y_train = []

for (patter_sentence, tag) in xy:
    bag = bag_of_words(patter_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)


class ChatDataset(Dataset):
    def __init__(self):
        self.num_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.num_samples

dataset = ChatDataset()

train_loader = DataLoader(
    dataset = dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 0
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

file = "data.pth"
torch.save(data, file)

print(f'Training complete.\nFile saved to {file}')
