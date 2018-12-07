# import torch as t
#
#
# class TransformerSearcher:
#     def __init__(self, transformer_decoder, output_linear, embedding, sos_id, eos_id):
#         self.sos_id = sos_id
#         self.eos_id = eos_id
#         self.transformer_decoder = transformer_decoder
#         self.output_linear = output_linear
#         self.embedding = embedding
#         self.max_lenth = transformer_decoder.max_lenth
#
#     def get_masks(self, input, device, batch_size):
#         input_mask = input.data.ne(0)
#         self_attention_mask = t.bmm(input_mask.unsqueeze(-1), input_mask.unsqueeze(-2))
#         dot_attention_mask = t.bmm(input_mask.unsqueeze(-1), t.ones((batch_size, 1, 1), dtype=t.long, device=device))
#         return input_mask, self_attention_mask, dot_attention_mask
#
#
#     def beam_search(self, encoder_output, beam_size=5):
#         """
#         :param encoder_output: B, L, E
#         :return:
#         """
#         batch_size = encoder_output.size(0)
#         device = encoder_output.device
#         init_input_token = t.LongTensor([self.sos_id] * batch_size).unsqueeze(-1).to(device)
#
#         for step in range(self.max_lenth):
#             pass
#         return None
#
#
#     def greedy_search(self, encoder_output):
#         """
#
#         :param init_input: B, 1, E
#         :param encoder_output: B, 1, E
#         :return:
#         """
#         batch_size = encoder_output.size(0)
#         device = encoder_output.device
#         outputs = t.zeros((batch_size, self.max_lenth))
#         input = t.zeros((batch_size, self.max_lenth))
#         input[]
#         input = t.LongTensor([self.sos_id] * batch_size).unsqueeze(-1).to(device)
#
#         for step in range(self.max_lenth):
#             input_mask, self_attention_mask, dot_attention_mask = self.get_masks(input, device, batch_size)
#             input_embedding = self.embedding(input)
#             pass
#
#         return None
#
#
#
#
