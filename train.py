import torch
import torch.nn as nn
from torchvision import transforms

import sys
import argparse
import numpy as np

sys.path.insert(0, 'model')
sys.path.insert(0, 'dataloader')

from model.network import GRU
from dataloader.dataset import DataGenerator


def input2tensor(x, input_size):
    batch_size, seq_len = x.shape[0], x.shape[1]
    tensor = torch.zeros(batch_size, seq_len, input_size)
    for i in range(batch_size):
        for j in range(seq_len):
            tensor[i][j][x[i][j]] = 1
    # batch_size x seq_len x input_size --> seq_len x batch_size x input_size
    tensor = tensor.permute(1, 0, 2)
    return tensor


def target2tensor(target):
    tensor = torch.LongTensor(target)
    # batch_size x seq_len --> seq_len x batch_size
    tensor = tensor.permute((1, 0))
    return tensor


def accuracy(output, target):
    # print(output)
    output = np.argmax(output, axis=2)
    acc = np.equal(output, target)
    acc = np.mean(acc.astype(np.float))
    return acc


def train_gru(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = DataGenerator(seq_length=config.seq_len, batch_size=config.batch_size, training_file='data/belling_the_cat.txt')

    gru = GRU(train_dataset.vocab_size, config.hidden_size, train_dataset.vocab_size, n_layers=config.n_layers).to(device)

    criterion = nn.NLLLoss().to(device)
    optim = torch.optim.RMSprop([{'params': gru.parameters(), 'lr': config.lr}])

    hidden = gru.initHidden(batch_size=config.batch_size).to(device)
    for step in range(config.epochs):
        optim.zero_grad()
        # x_batches: batch_size x seq_len x input_size
        # y_batches: batch_size x seq_len
        x_batches, y_batches = train_dataset.next_batch()
        # x: seq_len x batch_size x input_size
        x = input2tensor(x_batches, train_dataset.vocab_size).to(device)
        # output: seq_len x batch_size x output_size
        output, hidden = gru(x, hidden)
        # target: seq_len x batch_size
        target = target2tensor((y_batches))

        loss = 0
        for i in range(config.seq_len):
            loss = loss + criterion(output[i], target[i])

        loss.backward()
        optim.step()

        if step % 100 == 0:
            print("loss === %.4f" % (loss.item() / config.seq_len))
            x_batches = x.detach().cpu().numpy().transpose((1, 0, 2))
            output = output.detach().cpu().numpy().transpose((1, 0, 2)) # batch_size x seq_len x output_size

            print("Accuracy: %.2f" % (accuracy(output, y_batches)))

            x_batches = ''.join([train_dataset.reverse_dictionary[np.argmax(x)] for x in x_batches[0]])
            predict = ''.join([train_dataset.reverse_dictionary[np.argmax(x)] for x in output[0]])
            target = ''.join([train_dataset.reverse_dictionary[x] for x in y_batches[0]])

            print('[input] %s -> [target] %s ========= [predict] %s' % (str(x_batches), str(target), str(predict)))

        hidden = hidden.detach()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=2)

    config = parser.parse_args()

    # train_rnn(config)
    train_gru(config)