import h5py
import spacy
from collections import Counter
from loaders import get_dataset
from tqdm import tqdm
import logging
import numpy as np
import pickle as pk
import gc


hdf_file = 'processed.hdf'

logger = logging.getLogger(__name__)


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
        self.logger = logging.getLogger('preprocess logger')

    def str2id(self, str):
        ids = [self._vocab_t2i[i.text.lower()] if i.text.lower() in self._vocab_t2i else self._vocab_t2i['<UNK>'] for i in self.nlp(str)]
        return ids

    def handle_vocab(self):
        train_set = get_dataset('train')
        self.logger.info('start handle vocab')
        counter = Counter()

        for instance in tqdm(train_set, desc='handling vocab'):
            captions = instance[1]
            for caption in captions:
                token_list = [i.text.lower() for i in self.nlp(caption)]
                counter.update(token_list)
        self.logger.info(f'num of vocab is {len(counter)}')
        for i in counter:
            index = len(self._vocab_i2t)
            self._vocab_i2t[index] = i
            self._vocab_t2i[i] = index
        logger.info(f'vocab built.')

    def process_train(self):
        self.logger.info('start process train_set')
        self.train_features = []
        self.train_labels = []
        self.train_lenths = []
        train_set = get_dataset('train')
        for instance in tqdm(train_set, desc='processing train set'):
            if not len(instance[1]) >= 5:
                continue
            captions = instance[1]
            captions = [[self._vocab_t2i['<BOS>']] + self.str2id(i) + [self._vocab_t2i['<EOS>']] for i in captions][:5]
            padded_captions = pad_sequences(captions, 50)
            captions_lenths = [len(i) for i in captions]

            for i in range(5):
                self.train_features.append(instance[0].numpy())
                self.train_labels.append(padded_captions[i])
                self.train_lenths.append(captions_lenths[i])

    def process_val(self):
        self.logger.info('start process val_set')
        self.val_features = []
        self.val_labels = []
        self.val_lenths = []
        val_set = get_dataset('val')
        for instance in tqdm(val_set, desc='processing val set'):
            assert len(instance[1]) >= 5
            captions = instance[1]
            captions = [[self._vocab_t2i['<BOS>']] + self.str2id(i) + [self._vocab_t2i['<EOS>']] for i in captions][:5]
            padded_captions = pad_sequences(captions, 50)
            captions_lenths = [len(i) for i in captions]
            self.val_features.append(instance[0].numpy())
            self.val_labels.append(padded_captions)
            self.val_lenths.append(captions_lenths)

    def save_to_hdf(self):
        with h5py.File('test.hdf', 'w') as file:
            train_group = file.create_group('train')
            val_group = file.create_group('val')
            meta_group = file.create_group('meta')
            train_group.create_dataset('feature', data=np.array(self.train_features))
            del self.train_features
            train_group.create_dataset('label', data=np.array(self.train_labels))
            del self.train_labels
            train_group.create_dataset('lenths', data=np.array(self.train_lenths))
            del self.train_lenths
            gc.collect()
            val_group.create_dataset('feature', data=np.array(self.val_features))
            del self.val_features
            val_group.create_dataset('label', data=np.array(self.val_labels))
            del self.val_labels
            val_group.create_dataset('lenths', data=np.array(self.val_lenths))
            del self.val_lenths
            gc.collect()
        vocab = {'t2i': self._vocab_t2i, 'i2t': self._vocab_i2t}
        pk.dump(vocab, open('vocab.pkl', 'wb'))


if __name__ == '__main__':
    process = Preprocess()
    process.handle_vocab()
    process.process_train()
    process.process_val()
    process.save_to_hdf()







