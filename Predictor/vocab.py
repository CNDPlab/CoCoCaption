import torch as t
from tqdm import tqdm
from collections import Counter
import pickle as pk
from configs import DefaultConfig
import ipdb


class BaseVocab(object):
    def __init__(self):
        self.init_token = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.counter = Counter()
        self.offset = len(self.init_token)
        self.token2id = {v: i for i, v in enumerate(self.init_token)}
        self.id2token = {i: v for i, v in enumerate(self.init_token)}

    def add_sentance_token(self, sentance_token):
        self.counter.update(sentance_token)

    def filter_rare_token_build_vocab(self, min_count=5):
        self.common_words = [i for i, v in list(filter(lambda x: x[1] > min_count, self.counter.items())) if i not in self.init_token]
        print(f'filtered {len(self.counter)-len(self.common_words)} words,{len(self.common_words)} left')
        for index, word in enumerate(self.common_words):
            self.token2id[word] = index + self.offset
            self.id2token[index+self.offset] = word

    def add_list_sentance_token(self, list_sentance_token):
        for sentance_token in tqdm(list_sentance_token):
            self.add_sentance_token(sentance_token)

    def convert(self, input, type):
        assert type in ['t2i', 'i2t', 'lt2i', 'li2t']
        if type == 't2i':
            return self._convert_token2id_sentance(input)
        if type == 'i2t':
            return self._convert_id2token_sentance(input)
        if type == 'lt2i':
            return self._convert_token2id_sentance_list(input)
        if type == 'li2t':
            return self._convert_id2token_sentance_list(input)

    def save(self, path):
        pk.dump(self, open(path, 'wb'))

    def random_init(self, embedding_dim):
        self.matrix = t.nn.Embedding(len(self.token2id), embedding_dim, padding_idx=self.token2id['<PAD>']).weight
        print(self.matrix.shape)

    def use_pretrained(self, pretrained_model):
        filtered_tokens = [i for i in self.token2id if i in pretrained_model.wv or i in self.init_token]
        self.oovs = [i for i in self.token2id if i not in pretrained_model.wv and i in self.init_token]
        print(f'origin vocab lenth: {len(self.token2id)}')

        print(f'oovs num :{len(self.oovs)}')
        self.token2id = {v: i for i, v in enumerate(filtered_tokens)}
        self.id2token = {i: v for i, v in enumerate(filtered_tokens)}
        print(f'vocab lenth:{len(self.token2id)}')
        self.matrix = t.nn.Embedding(len(self.token2id), pretrained_model.vector_size, padding_idx=self.token2id['<PAD>']).weight
        with t.no_grad():
            for i in tqdm(range(len(self.token2id))):
                if self.id2token[i] in pretrained_model.wv:
                    self.matrix[i, :] = t.Tensor(pretrained_model.wv[self.id2token[i]])

    def _convert_token2id(self, token):
        if token in self.token2id:
            return self.token2id[token]
        else:
            return self.token2id['<UNK>']

    def _convert_token2id_sentance(self, sentance_token):
        return [self._convert_token2id(word) for word in sentance_token]

    def _convert_token2id_sentance_list(self, list_sentance_token):
        return [self._convert_token2id_sentance(sentance_token) for sentance_token in list_sentance_token]

    def _convert_id2token(self, id):
        return self.id2token[id]

    def _convert_id2token_sentance(self, sentance_id):
        return [self._convert_id2token(id) for id in sentance_id]

    def _convert_id2token_sentance_list(self, list_sentance_id):
        return [self._convert_id2token_sentance(sentance_id) for sentance_id in list_sentance_id]

    @property
    def vocab_size(self):
        return self.matrix.shape[0]
    @property
    def embedding_dim(self):
        return self.matrix.shape[1]

def load_vocab():
    return pk.load(open(DefaultConfig.vocab, 'rb'))
