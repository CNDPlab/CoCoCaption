import torchvision as tv
import torch as t
import numpy as np
import torch.nn.functional as F
import ipdb
import math


class VGGTransformerNew(t.nn.Module):
    def __init__(self, vocab, args):
        super(VGGTransformerNew, self).__init__()
        self.args = args
        self.vocab = vocab

        self.vgg_feature = tv.models.vgg16(True).features
        self.vgg_feature.requires_grad = False
        self.vgg_input = t.nn.Sequential(*list(tv.models.vgg16(True).classifier.children())[:-1])

        self.vgg_input_reshape = t.nn.Sequential(
            Linear(4096, 512, args.dropout),
            t.nn.ReLU(True)
        )
        self.feature_size = 512
        self.word_embedding = t.nn.Embedding(vocab.matrix.shape[0], vocab.matrix.shape[1], padding_idx=0,
                                             _weight=vocab.matrix)
        self.input_reshape = Linear(512 * 2, args.embedding_size, args.dropout)
        self.transformer_decoder = TransformerDecoder(args.embedding_size, self.feature_size, args.hidden_size,
                                                      args.dropout, args.num_head, max_lenth=1+args.max_seq_len,
                                                      max_time=12)
        self.output_linear = t.nn.Linear(vocab.matrix.shape[1], vocab.matrix.shape[0], bias=False)
        #self.output_linear.weight = self.word_embedding.weight
        #self.output_scale = args.embedding_size ** -0.5

    def get_masks(self, batch_size, word_input, feature):
        device = word_input.device
        feature_lenth = feature.size(1)
        input_mask = word_input.data.ne(0)
        input_lenth = input_mask.size(1)
        feature_mask = t.ones((batch_size, feature_lenth), dtype=t.uint8, device=device)

        self_attention_mask = t.bmm(input_mask.float().unsqueeze(-1), input_mask.float().unsqueeze(-2)).byte()
        dot_attention_mask = t.bmm(input_mask.float().unsqueeze(-1), feature_mask.float().unsqueeze(-2)).byte()
        direction_mask = t.tril(t.ones((input_lenth, input_lenth), dtype=t.uint8, device=device))
        direction_mask = direction_mask.unsqueeze(0).expand(batch_size, -1, -1)
        return input_mask, self_attention_mask, dot_attention_mask, direction_mask

    def forward(self, image_tensor, captions):
        # prepare inputs
        if len(image_tensor.size()) == 5:
            image_tensor = image_tensor.view(-1, image_tensor.size(-3), image_tensor.size(-2), image_tensor.size(-1))
            captions = captions.view(-1, captions.size(-1))
        else:
            pass
        batch_size = image_tensor.size(0)
        image_feature = self.vgg_feature(image_tensor).permute(0, 2, 3, 1).contiguous().view(batch_size, -1,
                                                                                             self.args.embedding_size)
        image_input = self.vgg_input(image_feature.view(batch_size, -1))
        image_input = self.vgg_input_reshape(image_input).unsqueeze(1).expand(-1, captions.size(-1), -1)
        word_input = self.word_embedding(captions)

        transformer_input = t.cat([word_input, image_input], -1)
        transformer_input = self.input_reshape(transformer_input)

        # inference
        input_mask, self_attention_mask, dot_attention_mask, direction_mask = self.get_masks(batch_size, captions, image_feature)
        transformer_input *= input_mask.float().unsqueeze(-1)

        transformer_output, self_attention_matrix, dot_attention_matrix = self.transformer_decoder(
            transformer_input, image_feature, input_mask, self_attention_mask, dot_attention_mask, direction_mask
        )
        output_log_prob = self.output_linear(transformer_output) #* self.output_scale
        output_token = output_log_prob.argmax(-1)
        return output_log_prob, output_token

    def greedy_search(self, image_tensor):
        batch_size = image_tensor.size(0)
        device = image_tensor.device

        for i in range(self.args.max_seq_len):
            if i == 0:
                input_caption = t.ones((batch_size, 1), dtype=t.long, device=device) * self.vocab.token2id['<BOS>']
            else:
                input_caption = t.cat([input_caption, output_token[:, -1:]], -1)
            output_log_prob, output_token = self.forward(image_tensor, input_caption)

        return output_token

def Linear(in_features, out_features, dropout=0.):
    m = t.nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return t.nn.utils.weight_norm(m)


class TransformerDecoder(t.nn.Module):
    def __init__(self, embedding_size, encoder_output_size, hidden_size, dropout, num_head, max_lenth=25, max_time=6):
        super(TransformerDecoder, self).__init__()
        self.position_embedding = PositionEncoding(max_lenth=max_lenth + 5, embedding_size=embedding_size)
        self.time_embedding = PositionEncoding(max_lenth=max_time + 5, embedding_size=embedding_size)
        self.decoder_block_list = t.nn.ModuleList(
            [TransformerDecoderBlock(embedding_size, encoder_output_size, hidden_size, dropout, num_head) for i in range(6)]
        )
        self.layer_norm = t.nn.LayerNorm(embedding_size)
        self.max_batch_size = 256
        self.max_time = max_time
        self.max_lenth = max_lenth
        self._init_position_feature()

    def _init_position_feature(self):
        self.positions = t.range(1, self.max_lenth+1).repeat(self.max_batch_size).view(self.max_batch_size, self.max_lenth+1).long()

    def get_position_feature(self, batch_size, seq_lenth, device):
        return self.positions[:batch_size, :seq_lenth].data.to(device)

    def forward(self, word_embedding, encoder_output, input_mask, self_attention_mask=None, dot_attention_mask=None,
                direction_mask=None):
        if self_attention_mask is not None and direction_mask is not None:
            self_attention_direction_mask = self_attention_mask * direction_mask
        elif self_attention_mask is None and direction_mask is not None:
            self_attention_direction_mask = self_attention_mask
        elif self_attention_mask is not None and direction_mask is None:
            self_attention_direction_mask = self_attention_mask
        else:
            self_attention_direction_mask = None

        batch_size, seq_lenth, embedding_size = word_embedding.size()
        device = word_embedding.device
        position_feature = self.get_position_feature(batch_size, seq_lenth, device)
        position_feature *= input_mask.long()
        position_embedding = self.position_embedding(position_feature)
        self_attention_matrixs = {}
        dot_attention_matrixs = {}
        for step, block in enumerate(self.decoder_block_list):
            if step == 0:
                embedding = word_embedding + position_embedding
            else:
                embedding = embedding + position_embedding
            embedding, self_attention_matrix, dot_attention_matrix = block(embedding,
                                                                           encoder_output,
                                                                           input_mask,
                                                                           self_attention_direction_mask,
                                                                           dot_attention_mask)
            self_attention_matrixs[step] = self_attention_matrix
            dot_attention_matrixs[step] = dot_attention_matrix
        return embedding, self_attention_matrixs, dot_attention_matrixs


class TransformerDecoderBlock(t.nn.Module):
    def __init__(self, embedding_size, encoder_output_size, hidden_size, dropout, num_head):
        super(TransformerDecoderBlock, self).__init__()
        self.multi_head_self_attention = MultiHeadAttention(num_head, hidden_size, embedding_size,
                                                            embedding_size, embedding_size, dropout, embedding_size)
        self.dropout1 = t.nn.Dropout(dropout)
        self.layer_norm1 = t.nn.LayerNorm(embedding_size)
        self.multi_head_dot_attention = MultiHeadAttention(num_head, hidden_size, embedding_size,
                                                           encoder_output_size, encoder_output_size, dropout, embedding_size)
        self.dropout2 = t.nn.Dropout(dropout)
        self.layer_norm2 = t.nn.LayerNorm(embedding_size)
        self.transition = FeedForward(embedding_size, dropout)
        self.dropout3 = t.nn.Dropout(dropout)
        self.layer_norm3 = t.nn.LayerNorm(embedding_size)

    def forward(self, embedding, encoder_output, input_mask, self_attention_mask=None, dot_attention_matrix=None):
        input_mask = input_mask.float().unsqueeze(-1)
        res1 = embedding
        net, self_attention_matrix = self.multi_head_self_attention(embedding, embedding, embedding, self_attention_mask)
        net += res1
        net = self.dropout1(net)
        net = self.layer_norm1(net)
        net *= input_mask
        res2 = net
        net, dot_attention_matrix = self.multi_head_dot_attention(net, encoder_output, encoder_output, dot_attention_matrix)
        net += res2
        net = self.dropout2(net)
        net = self.layer_norm2(net)
        net *= input_mask
        res3 = net
        net = self.transition(net)
        net += res3
        net = self.dropout3(net)
        net = self.layer_norm3(net)
        net *= input_mask
        return net, self_attention_matrix, dot_attention_matrix


class MultiHeadAttention(t.nn.Module):
    def __init__(self, num_head, hidden_size, query_dim, key_dim, value_dim, dropout, output_dim):
        super(MultiHeadAttention, self).__init__()
        self.dropout = t.nn.Dropout(dropout)
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.key_projection = t.nn.Linear(key_dim, self.num_head * self.hidden_size, bias=False)
        self.query_projection = t.nn.Linear(query_dim, self.num_head * self.hidden_size, bias=False)
        self.value_projection = t.nn.Linear(value_dim, self.num_head * self.hidden_size, bias=False)
        self.scale = np.sqrt(self.hidden_size)
        self.linear = t.nn.Linear(self.num_head * self.hidden_size, output_dim, bias=False)
        t.nn.init.xavier_normal_(self.key_projection.weight)
        t.nn.init.xavier_normal_(self.query_projection.weight)
        t.nn.init.xavier_normal_(self.value_projection.weight)
        t.nn.init.xavier_normal_(self.linear.weight)

    def forward(self, query, key, value, attention_mask=None):
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)
        # key = value
        batch_size, query_lenth, query_dim = query.size()
        key_lenth = key.size(1)
        query_projection = self.query_projection(query).view(batch_size, query_lenth, self.num_head, self.hidden_size).permute(0, 2, 1, 3)
        # B, N, QL, H

        key_projection = self.key_projection(key).view(batch_size, key_lenth, self.num_head, self.hidden_size).permute(0, 2, 3, 1)
        # B, N, H, KL

        value_projection = self.value_projection(value).view(batch_size, key_lenth, self.num_head, self.hidden_size).permute(0, 2, 1, 3)
        # B, N, KL, H

        attention_matrix = query_projection @ key_projection
        # B, N, QL, KL

        if attention_mask is not None:
            attention_matrix.masked_fill_(attention_mask == 0, -float('inf'))

        attention_matrix = F.softmax(attention_matrix, -1)
        attention_matrix = attention_matrix.masked_fill(t.isnan(attention_matrix), 0)
        attention_matrix = self.dropout(attention_matrix)
        weighted = attention_matrix @ value_projection
        # B, N, QL, KL * B, N, KL, H -> B, N，QL, H
        output = weighted.permute(0, 2, 1, 3).contiguous().view(batch_size, query_lenth, self.num_head * self.hidden_size)
        output = self.linear(output)
        return output, attention_matrix


class FeedForward(t.nn.Module):
    def __init__(self, input_size, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = t.nn.Conv1d(input_size, input_size * 2, 1)
        self.linear2 = t.nn.Conv1d(input_size * 2, input_size, 1)
        self.relu = t.nn.ReLU(True)
        t.nn.init.xavier_normal_(self.linear1.weight)
        t.nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, inputs):
        net = self.linear1(inputs.transpose(1, 2))
        net = self.relu(net)
        net = self.linear2(net)
        net = net.transpose(1, 2)
        return net


class PositionEncoding(t.nn.Module):
    def __init__(self, max_lenth, embedding_size):
        super(PositionEncoding, self).__init__()
        self.max_lenth = max_lenth + 1
        self.embedding_size = embedding_size
        self.position_encoding = t.nn.Embedding(max_lenth, embedding_size, padding_idx=0)
        self.init()

    def init(self):
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / self.embedding_size) for j in range(self.embedding_size)] if pos != 0
             else np.zeros(self.embedding_size) for pos in range(self.max_lenth + 1)])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        self.position_encoding.weight.data = t.from_numpy(position_enc).float()
        self.position_encoding.weight.requires_grad = False

    def forward(self, position_feature):
        # inputs [B, max_lenth]
        positions_encoded = self.position_encoding(position_feature)
        return positions_encoded


if __name__ == '__main__':
    import ipdb
    from loaders import get_loader
    from configs_transformer import DefaultConfig
    from tqdm import tqdm
    from vocabulary import Vocab
    args = DefaultConfig
    args.batch_size = 2
    loader = get_loader('train', args.batch_size)
    vocab = Vocab()

    for i in tqdm(loader):
        feature, captions = [j for j in i]
        model = VGGTransformerNew(vocab, args)
        output_log_prob, output_token = model(feature, captions.long())
        token = model.greedy_search(feature[:, 0])
        loss = output_log_prob.sum()
        loss.backward()
        d = []
        for i in model.named_parameters():
            if i[0][:3] != 'vgg':
                print(i[0])
                print(i[1].grad)
                input('next')


