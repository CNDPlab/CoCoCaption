import h5py
import spacy
from collections import Counter
from loaders import get_dataset
from tqdm import tqdm
import logging
import numpy as np
import pickle as pk
import os
import gc
import shutil

## TODO
"""












"""
hdf_file = 'processed.hdf'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import ipdb


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='post', truncating='post', value=0.):
    """
    FROM KERAS
    Pads each sequence to the same length:
    the length of the longest sequence.
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


class Preprocess:
    def __init__(self):
        self.hdf_file = hdf_file
        self.nlp = spacy.load('en')
        self.embedding_dim = 300
        self.init_token = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self._vocab_t2i = {v: i for i, v in enumerate(self.init_token)}
        self._vocab_i2t = {i: v for i, v in enumerate(self.init_token)}
        with h5py.File('processed.hdf', 'w') as file:
            train = file.create_group('train')
            val = file.create_group('val')
            train.create_dataset('feature', (1, 3, 224, 224), maxshape=(None, 3, 224, 224), compression='gzip')
            train.create_dataset('label', (1, 5, 65), maxshape=(None, 5, 65), compression='gzip')
            train.create_dataset('lenth', (1, 5), maxshape=(None, 5), compression='gzip')

            val.create_dataset('feature', (1, 3, 224, 224), maxshape=(None, 3, 224, 224), compression='gzip')
            val.create_dataset('label', (1, 5, 65), maxshape=(None, 5, 65), compression='gzip')
            val.create_dataset('lenth', (1, 5), maxshape=(None, 5), compression='gzip')

    def str2id(self, str):
        ids = [self._vocab_t2i[i.text.lower()] if i.text.lower() in self._vocab_t2i else self._vocab_t2i['<UNK>'] for i in self.nlp(str)]
        return ids

    def handle_vocab(self):
        train_set = get_dataset('train')
        counter = Counter()

        for instance in tqdm(train_set, desc='handling vocab'):
            captions = instance[1]
            for caption in captions:
                token_list = [i.text.lower() for i in self.nlp(caption)]
                counter.update(token_list)
        for i in counter.items():
            if i[1] >= 5:
                index = len(self._vocab_i2t)
                self._vocab_i2t[index] = i[0]
                self._vocab_t2i[i[0]] = index
        vocab = {'t2i': self._vocab_t2i, 'i2t': self._vocab_i2t}
        pk.dump(vocab, open('vocab.pkl', 'wb'))

    def process_train(self):
        self.writer = h5py.File('processed.hdf', 'a')
        train_set = get_dataset('train')
        for index, instance in tqdm(enumerate(train_set), desc='processing train set'):
            if not len(instance[1]) >= 5:
                continue
            captions = instance[1]
            captions = [[self._vocab_t2i['<BOS>']] + self.str2id(i) + [self._vocab_t2i['<EOS>']] for i in captions][:5]
            padded_captions = pad_sequences(captions, 65)
            captions_lenths = [len(i) for i in captions]

            self.writer['train']['feature'].resize([index + 1, 3, 224, 224])
            self.writer['train']['feature'][index:index + 1] = instance[0].numpy()
            self.writer['train']['label'].resize([index + 1, 5, 65])
            self.writer['train']['label'][index:index + 1] = padded_captions
            self.writer['train']['lenth'].resize([index + 1, 5, ])
            self.writer['train']['lenth'][index:index + 1] = captions_lenths
            gc.collect()
        self.writer.close()

    def process_val(self):
        self.writer = h5py.File('processed.hdf', 'a')
        val_set = get_dataset('val')
        for index, instance in tqdm(enumerate(val_set), desc='processing val set'):
            assert len(instance[1]) >= 5
            captions = instance[1]
            captions = [[self._vocab_t2i['<BOS>']] + self.str2id(i) + [self._vocab_t2i['<EOS>']] for i in captions][:5]
            padded_captions = pad_sequences(captions, 65)
            captions_lenths = [len(i) for i in captions]
            self.writer['val']['feature'].resize([index + 1, 3, 224, 224])
            self.writer['val']['feature'][index:index + 1] = instance[0].numpy()
            self.writer['val']['label'].resize([index + 1, 5, 65])
            self.writer['val']['label'][index:index + 1] = padded_captions
            self.writer['val']['lenth'].resize([index + 1, 5, ])
            self.writer['val']['lenth'][index:index + 1] = captions_lenths
        gc.collect()
        self.writer.close()


if __name__ == '__main__':
    process = Preprocess()
    process.handle_vocab()
    process.process_train()
    process.process_val()





# file = h5py.File('processed.hdf', 'r')
# file['train']['feature][10].shape

