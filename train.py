
import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load intents file
with open("E:\\chatbot-Ai240\\Data.json", "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Loop through each sentence in our intents patterns
for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", ".", "!"]  # these words they wont be as a unique words
all_words = [stem(w) for w in all_words if w not in ignore_words]
# it will iterate in each word 'w' in the all_Words list it will check
# if the word is not in the ignore_words list and if its not in the ignore it will stemm it
all_words = sorted(set(all_words))
# after filtering that the word isnt in the ignored it convert the list to set to prevent the duplicates
tags = sorted(set(tags))

# Prepare training data
X_train = []
y_train = []
# xy they will populated with the training data
for pattern_sentence, tag in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # . This vector represents the input data for training the chatbot model.
    label = tags.index(tag)
    # It essentially maps the intent tag to a numerical label.
    y_train.append(label)
# This list contains the corresponding labels (intent tags) for the training data.
X_train = np.array(X_train)
y_train = np.array(y_train)
# after iterating on all training data we will convert x and y to numerical arrays


# Model parameters

# the number of queries that that the model trained on
num_epochs = 1000
# batch size is the number of the training examples at a time:
# its configuration setting for the training proccess on the data
batch_size = 8

# means smaller updates and potentially slower but more stable training.
learning_rate = 0.001

# it's determined by the length of a single bag-of-words vector
# which is the same as the number of unique words in the vocabulary
input_size = len(X_train[0])

# it determines the complexity and
# capacity of the model to capture patterns in the data.
hidden_size = 8

# represents the number of output units in the neural network's output layer.
output_size = len(tags)


# Dataset preparation
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # will iterate on speceified number of epochs
    # one epoch represent complete pass through the entire training data

    # train loader: iterate through each mini-btaches of the training data
    # train loader collect the training data
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        # 2 lines before : transfer the mini batch to
        # the gpu if avaialbe other wise "cpu" for speeding the process

        # t computes predictions for the current mini-batch of input data
        outputs = model(words)

        # calculates the loss or error between the model's predictions
        # and the true labels for mini-batch
        loss = criterion(outputs, labels)

        # clears the gradients of the model's
        # parameters to prepare for backpropagation.
        optimizer.zero_grad()

        # computes the gradients of the loss
        loss.backward()

        # . This step is responsible
        # for training the model and improving its performance.
        optimizer.step()

    # This is useful for monitoring training.
        # After processing all mini-batches within an epoch
        # the loop will go tho the next epoch unyil the specfied number
        # of epoch is reached
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # After all epochs are completed, the final loss for the last epoch
        # is printed to provide the insights.
print(f"final loss: {loss.item():.4f}")


# Save trained model

# making it possible to load and use the model for making predictions or
# responding to user queries in a chatbot application
# without having to retrain the model from scratch.
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}

FILE = "data.pth"
torch.save(data, FILE)
print(f"training complete. file saved to {FILE}")
