import torch
import re
import spacy
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

# Check first 5 examples
for i, example in enumerate(dataset):
    print(example["text"])
    if i >= 4:
        break

#text loading
text = example["text"]

# Keep only letters and numbers, replace others with space
cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
# Convert multiple spaces to single space
cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
# Lowercase
cleaned_text = cleaned_text.lower()

#tokenization
nlp = spacy.blank('en')
doc = nlp(cleaned_text)
vocab = []
for token in doc:
  vocab.append(token.text)

word_2_idx = {"<UNK>": 0}  # unknown token
for idx, word in enumerate(set(vocab)):
    word_2_idx[word] = idx + 1

X = []
Y = []

for i in range(len(vocab) - 1):
    X.append(word_2_idx[vocab[i]])     # current word index
    Y.append(word_2_idx[vocab[i + 1]]) # next word index

x = torch.tensor(X)
y = torch.tensor(Y)

#custom dataset
class CustomDataset(Dataset):
  def __init__(self, features, labels):
    self.features = features
    self.labels = labels

  def __len__(self):
    return len(self.features)

  def __getitem__(self, idx):
    return self.features[idx], self.labels[idx]

train_dataset = CustomDataset(x, y)

class neural_network(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        num_embeddings = len(word_2_idx) + 1
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_embeddings)  # predict next word

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return x

#objective
def objective(trial):
  epochs = trial.suggest_int('epochs', 1, 70)
  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log = True)
  Weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e6, log = True)
  Batch = trial.suggest_int('Batch', 1,100)
  embedding_dim = trial.suggest_int('embedding_dim', 512, 1024)

  train_loader = DataLoader(train_dataset, batch_size = Batch, shuffle = True)
  model = neural_network(embedding_dim)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = Weight_decay)

  for epoch in range(epochs):
    total_epoch_loss = 0
    for batch_features, batch_labels in train_loader:#after put here test_loader
      optimizer.zero_grad()
      outputs = model(batch_features)
      loss = criterion(outputs, batch_labels)
      loss.backward()
      optimizer.step()
      total_epoch_loss += loss.item()

  model.eval()

  total = 0
  correct = 0
  with torch.no_grad():
    for batch_features, batch_labels in train_loader:
      outputs = model(batch_features)
      _, predicted = torch.max(outputs.data, 1)
      total += batch_labels.size(0)
      correct += (predicted == batch_labels).sum().item()

  accuracy = 100 * correct / total
  return accuracy

import optuna
study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 50)
print("Best value is : ",study.best_value)
