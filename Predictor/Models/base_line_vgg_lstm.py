import torch as t
import torchvision as tv
from Predictor.Utils import show_img
from Predictor.Models import BaseModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import ipdb
from Predictor.vocab import load_vocab



class BaseLine(BaseModel):
    def __init__(self, vocab, args):
        super(BaseLine, self).__init__(vocab, args)
        self.BOSid = t.LongTensor(vocab._convert_token2id('<BOS>')).cuda()
        self.lstm = t.nn.LSTM(self.args.embedding_size, self.args.hidden_size, self.args.num_layers, batch_first=True)
        self.dense = t.nn.Linear(512*7*7, self.args.embedding_size)
        self.linear = t.nn.Linear(self.args.hidden_size, self.vocab.matrix.shape[0])
        self.max_seq_len = self.args.max_seq_len


    def forward(self, img_tensor, captions, lengths, tru=None):
        features = self.vgg(img_tensor)
        features = self.dense(features.view(features.shape[0], -1))
        embedding = self.word_embedding(captions.long().cuda())
        embedding = t.cat((features.unsqueeze(1), embedding), 1)
        packed = pack_padded_sequence(embedding, lengths, batch_first=True)
        outpack, _ = self.lstm(packed)
        out = pad_packed_sequence(outpack, batch_first=True)
        outputs = self.linear(out[0])
        return outputs

    def build_decoder(self):
        pass

    def greedy_search(self, features):
        ids = []
        inputs = self.vgg(features)
        inputs = self.dense(inputs.view(inputs.shape[0], -1))
        # hidden = t.FloatTensor([[0]*self.args.hidden_size]*inputs.shape[0]).cuda()
        # states = hidden
        # inputs = self.word_embedding(t.LongTensor([self.BOSid])*features.shape[0])
        for i in range(self.max_seq_len):
            if i == 0:
                hidden, states = self.lstm(inputs.unsqueeze(1))
            else:
                hidden, states = self.lstm(inputs.unsqueeze(1), states)
            outputs = self.linear(hidden.squeeze(1))
            # probs.append(outputs)
            _, predicted = outputs.max(1)
            ids.append(predicted)
            inputs = self.word_embedding(predicted)
        ids = t.stack(ids, 1)
        # probs = t.stack(probs, 1)
        return ids

    def beam_search(self):
        pass



if __name__ == '__main__':
    import ipdb
    from loaders import get_loader
    from configs import DefaultConfig
    args = DefaultConfig
    args.batch_size=3
    loader = get_loader('val', args.batch_size)
    vocab = load_vocab()

    for i in loader:
        feature, caption = [j for j in i]
        lengths = caption.ne(0).sum(-1).long()
        lengths, sort_ind = lengths.sort(dim=0, descending=True)
        feature = feature[sort_ind]
        caption = caption[sort_ind]
        model = BaseLine(vocab, args)
        model.greedy_search(feature)
        model(feature, caption, lengths)


