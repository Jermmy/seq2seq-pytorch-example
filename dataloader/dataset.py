import numpy as np


class DataGenerator:

    def __init__(self, seq_length, batch_size, training_file):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.content = self.read_data(training_file)
        self.dictionary, self.reverse_dictionary = self.build_dataset()
        self.vocab_size = len(self.dictionary)
        self._pointer = 0
        self.total_len = len(self.content)

        print("vocab_size " + str(self.vocab_size))

    def read_data(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        return content

    def build_dataset(self):
        chars = sorted(list(set(self.content)))
        dictionary = dict((c, i) for i, c in enumerate(chars))
        reverse_dictionary = dict((i, c) for i, c in enumerate(chars))
        return dictionary, reverse_dictionary

    def char2id(self, c):
        return self.dictionary[c]

    def id2char(self, id):
        return self.reverse_dictionary[id]

    def next_batch(self):
        x_batches = []
        y_batches = []
        for i in range(self.batch_size):
            if self._pointer + self.seq_length + 1 >= self.total_len:
                self._pointer = 0
            bx = self.content[self._pointer: self._pointer + self.seq_length]
            by = self.content[self._pointer + 1: self._pointer + self.seq_length + 1]
            self._pointer += self.seq_length

            bx = [[self.dictionary[c]] for c in bx]
            by = [self.dictionary[c] for c in by]

            x_batches += [bx]
            y_batches += [by]

        x_batches = np.array(x_batches)
        y_batches = np.array(y_batches)

        return x_batches, y_batches
