import torch as t
import numpy as np
import torch.nn.functional as F
import ipdb


###################################################################################
#### waring modules has a bit change for the first input is the picture feature####
###################################################################################

class TransformerEncoder(t.nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout, num_head, max_lenth=25, max_time=6):
        super(TransformerEncoder, self).__init__()
        self.position_embedding = PositionEncoding(max_lenth=max_lenth,embedding_size=embedding_size)
        self.time_embedding = PositionEncoding(max_lenth=max_time, embedding_size=embedding_size)
        self.encoder_block = TransformerEncoderBlock(embedding_size, hidden_size, dropout, num_head)
        self.layer_norm = t.nn.LayerNorm(embedding_size)
        self.max_batch_size = 128
        self.max_time = max_time
        self.max_lenth = max_lenth
        self._init_position_feature()
        self._init_time_feature()

    def _init_position_feature(self):
        self.positions = t.range(1, self.max_lenth).repeat(self.max_batch_size).view(self.max_batch_size, self.max_lenth).long()

    def _init_step_feature(self):
        self.step_features = {}
        for i in range(self.max_time):
            self.step_features[i] = t.ones((self.max_batch_size, self.max_lenth)).long() + (i + 1)

    def get_position_feature(self, batch_size, seq_lenth, device):
        return self.positions[:batch_size, :seq_lenth].data.to(device)

    def get_time_feature(self, batch_size, seq_lenth, device, time):
        return self.step_features[time][:batch_size, seq_lenth].data.to(device)

    def forward(self, word_embedding, input_mask, self_attention_mask=None):
        batch_size, seq_lenth, embedding_size = word_embedding.size()
        device = word_embedding.device
        position_embedding = self.get_position_feature(batch_size, seq_lenth, device)
        self_attention_matrixs = {}
        for step in range(self.max_time):
            time_embedding = self.get_time_feature(batch_size, seq_lenth, device, step)

            embedding = (word_embedding + position_embedding + time_embedding) * input_mask.float().unsqueeze(-1)
            embedding, self_attention_matrix = self.encoder_block(embedding, self_attention_mask)
            self_attention_matrix[step] = self_attention_matrix
        return embedding, self_attention_matrixs


class TransformerEncoderBlock(t.nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout, num_head):
        super(TransformerEncoderBlock, self).__init__()
        self.multi_head_self_attention = MultiHeadAttention(num_head, hidden_size, embedding_size,
                                                            embedding_size, embedding_size, dropout, embedding_size)
        self.dropout1 = t.nn.Dropout(dropout)
        self.layer_norm1 = t.nn.LayerNorm(embedding_size)
        self.transition = FeedForward(embedding_size, dropout)
        self.dropout2 = t.nn.Dropout(dropout)
        self.layer_norm2 = t.nn.LayerNorm(embedding_size)

    def forward(self, embedding, self_attention_mask=None):
        res1 = embedding
        net, self_attention_matrix = self.multi_head_self_attention(embedding, embedding, embedding, self_attention_mask)
        net = net + res1
        net = self.dropout1(net)
        net = self.layer_norm1(net)
        res2 = net
        net = self.transition(net)
        net = net + res2
        net = self.dropout2(net)
        net = self.layer_norm2(net)
        return net, self_attention_matrix
    

class TransformerDecoder(t.nn.Module):
    def __init__(self, embedding_size, encoder_output_size, hidden_size, dropout, num_head, max_lenth=25, max_time=6):
        super(TransformerDecoder, self).__init__()
        self.position_embedding = PositionEncoding(max_lenth=max_lenth + 5, embedding_size=embedding_size)
        self.time_embedding = PositionEncoding(max_lenth=max_time + 5, embedding_size=embedding_size)
        self.decoder_block = TransformerDecoderBlock(embedding_size, encoder_output_size, hidden_size, dropout, num_head)
        self.layer_norm = t.nn.LayerNorm(embedding_size)
        self.max_batch_size = 128
        self.max_time = max_time
        self.max_lenth = max_lenth
        self._init_position_feature()
        self._init_step_feature()

    def _init_position_feature(self):
        self.positions = t.range(1, self.max_lenth+1).repeat(self.max_batch_size).view(self.max_batch_size, self.max_lenth+1).long()

    def _init_step_feature(self):
        self.step_features = {}
        for i in range(self.max_time):
            self.step_features[i] = t.ones((self.max_batch_size, self.max_lenth+1)).long() + (i + 1)

    def get_position_feature(self, batch_size, seq_lenth, device):
        return self.positions[:batch_size, :seq_lenth].data.to(device)

    def get_time_feature(self, batch_size, seq_lenth, device, time):
        return self.step_features[time][:batch_size, :seq_lenth].data.to(device)

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
        for step in range(self.max_time):

            time_feature = self.get_time_feature(batch_size, seq_lenth, device, step)
            time_feature *= input_mask.long()
            time_embedding = self.time_embedding(time_feature)
            if step == 0:
                embedding = word_embedding + position_embedding + time_embedding
                embedding = self.layer_norm(embedding)
            else:
                embedding = embedding + position_embedding + time_embedding
                embedding = self.layer_norm(embedding)
            embedding, self_attention_matrix, dot_attention_matrix = self.decoder_block(embedding, encoder_output,
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

    def forward(self, embedding, encoder_output, self_attention_mask=None, dot_attention_matrix=None):
        res1 = embedding
        net, self_attention_matrix = self.multi_head_self_attention(embedding, embedding, embedding, self_attention_mask)
        net += res1
        net = self.dropout1(net)
        net = self.layer_norm1(net)
        res2 = net
        net, dot_attention_matrix = self.multi_head_dot_attention(net, encoder_output, encoder_output, dot_attention_matrix)
        net += res2
        net = self.dropout2(net)
        net = self.layer_norm2(net)
        res3 = net
        net = self.transition(net)
        net += res3
        net = self.dropout3(net)
        net = self.layer_norm3(net)
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
        self.drop = t.nn.Dropout(dropout)
        self.relu = t.nn.ReLU()
        t.nn.init.kaiming_normal_(self.linear1.weight)
        t.nn.init.kaiming_normal_(self.linear2.weight)

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
        position_enc = np.array([[(pos - 1) / np.power(10000, 2 * (j // 2)/self.embedding_size) for j in range(self.embedding_size)] if pos not in [0, 1]
                                 else np.zeros(self.embedding_size) for pos in range(self.max_lenth+1)])
        position_enc[2:, 0::2] = np.sin(position_enc[2:, 0::2])  # dim 2i
        position_enc[2:, 1::2] = np.cos(position_enc[2:, 1::2])  # dim 2i+1
        self.position_encoding.weight.data = t.from_numpy(position_enc).float()
        self.position_encoding.weight.requires_grad = False

    def forward(self, position_feature):
        # inputs [B, max_lenth]
        positions_encoded = self.position_encoding(position_feature)
        return positions_encoded

