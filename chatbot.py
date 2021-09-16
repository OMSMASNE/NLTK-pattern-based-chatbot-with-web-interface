# Copyright (c) 2021 OM SANTOSHKUMAR MASNE.
# All Rights Reserved.
# Licensed under the MIT license.
# See LICENSE file in the project root for license information.

import torch

import json
import random

from model import NeuralNet
from utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

data_file = "data.pth"
model_data = torch.load(data_file)

model_state = model_data["model_state"]
input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
all_words = model_data["all_words"]
tags = model_data["tags"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_answer(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, prediction = torch.max(output, dim = 1)

    tag = tags[prediction.item()]

    probs = torch.softmax(output, dim = 1)
    prob = probs[0][prediction.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return random.choice(intent['responses'])
    else:
        return "I do not understand that..."

def dry_run():
    while True:
        sentence = input("You: ")

        if sentence == "quit":
            break
        else:
            answer = get_answer(sentence)

        print("Bot:", answer)

if __name__ == "__main__":
    dry_run()
