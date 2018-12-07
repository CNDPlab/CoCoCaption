import torch as t
import torch.nn as nn
import torchvision as tv
from Predictor.Utils import show_img
from Predictor.Models import BaseModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import ipdb


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class BaseAttention(BaseModel):
    def __init__(self, vocab, args):
        super(BaseAttention, self).__init__(vocab, args)
        # self.BOSid = t.LongTensor(vocab._convert_token2id('<BOS>')).cuda()
        # self.lstm = t.nn.LSTM(self.args.embedding_size, self.args.hidden_size, self.args.num_layers, batch_first=True)
        self.decode_step = nn.LSTMCell(self.args.embedding_size + self.args.encoder_dim, self.args.hidden_size, bias=True)
        self.linear = t.nn.Linear(self.args.hidden_size, self.vocab.matrix.shape[0])
        self.init_h = nn.Linear(self.args.encoder_dim, self.args.hidden_size)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(self.args.encoder_dim, self.args.hidden_size)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(self.args.hidden_size, self.args.encoder_dim)
        self.max_seq_len = self.args.max_seq_len
        self.attention = Attention(self.args.encoder_dim, self.args.hidden_size, self.args.attention_dim)
        self.sigmoid = nn.Sigmoid()

    def build_decoder(self):
        pass

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, img_tensor, captions, lengths, tru=None):
        features = self.vgg2(img_tensor)
        batch_size = features.shape[0]
        features = features.view(batch_size, -1, self.args.encoder_dim)
        num_pixels = features.shape[1]
        embedding = self.word_embedding(captions.long().cuda())
        h, c = self.init_hidden_state(features)
        decode_lengths = (lengths - 1).tolist()
        probs = t.zeros(batch_size, max(decode_lengths), self.vocab.matrix.shape[0]).cuda()
        preds = t.zeros(batch_size, max(decode_lengths), num_pixels).cuda()
        for i in range(max(decode_lengths)):
            batch_size_t = sum([l > i for l in decode_lengths])
            attention_weighted_encoding, pred = self.attention(features[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                t.cat([embedding[:batch_size_t, i, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, hidden_size)
            prob = self.linear(h)
            probs[:batch_size_t, i, :] = prob
            preds[:batch_size_t, i, :] = pred
        return probs

    def greedy_search(self, features):
        ids = []
        features = self.vgg2(features)
        batch_size = features.shape[0]
        features = features.view(batch_size, -1, self.args.encoder_dim)

        for i in range(self.max_seq_len):
            if i == 0:
                hidden, states = self.lstm(features.unsqueeze(1))
            else:
                hidden, states = self.lstm(features.unsqueeze(1), states)
            outputs = self.linear(hidden.squeeze(1))
            _, predicted = outputs.max(1)
            ids.append(predicted)
            inputs = self.word_embedding(predicted)
        ids = t.stack(ids, 1)
        return ids

    def beam_search(self):
        pass


