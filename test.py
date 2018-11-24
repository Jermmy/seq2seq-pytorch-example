from dataloader.dataset import DataGenerator

import torch.nn as nn
import torch

g = DataGenerator(seq_length=2, batch_size=2, training_file='data/JayLyrics.txt')

embedding = nn.Embedding(g.vocab_size, 10)

x, y = g.next_batch()

x = torch.LongTensor(x)

print(x)
print(x.shape)

x = embedding(x)
print(x)
print(x.shape)

x = x.squeeze(2)
print(x)
print(x.shape)