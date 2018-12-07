import gensim
from loaders import get_raw_loader
from Predictor.vocab import BaseVocab
from tqdm import tqdm
from configs_transformer import DefaultConfig
from nltk.tokenize import word_tokenize
import ipdb


vocab = BaseVocab()

train_loader = get_raw_loader('train', 1)

for instance in tqdm(train_loader, desc='building vocab.'):
    img = instance[0]
    captions = instance[1]
    for caption in captions:
        vocab.add_sentance_token(['<BOS>']+[word.lower() for word in word_tokenize(caption[0])]+['<EOS>'])


vocab.filter_rare_token_build_vocab(5)
vocab.random_init(DefaultConfig.embedding_size)
print(vocab.matrix.shape)
vocab.save(DefaultConfig.vocab)






