from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_util import config
from numpy import random

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

    #seq_lens should be in descending order
    def forward(self, input, seq_lens):
        embedded = self.embedding(input)

        embedded_flat = embedded.view(-1, embedded.shape[2], embedded.shape[3])

        seq_lens_tensor = torch.LongTensor(seq_lens)
        if use_cuda:
            seq_lens_tensor = seq_lens_tensor.cuda()

        sorted_seq_lens, sorted_seq_lens_ind = seq_lens_tensor.view(-1).sort(descending=True)

        input_to_words_encoder = embedded_flat.clone()[sorted_seq_lens_ind, :, :]
        packed = pack_padded_sequence(input_to_words_encoder, sorted_seq_lens, batch_first=True)

        output, hidden = self.lstm(packed)

        output, _ = pad_packed_sequence(output, batch_first=True)  # h dim = B x t_k x n
        indxs_for_output = sorted_seq_lens_ind.unsqueeze(-1).unsqueeze(-1).expand(-1, output.shape[1],
                                                                                  output.shape[2])

        unsorted_output = torch.zeros_like(output)
        unsorted_output.scatter_(0, indxs_for_output, output)

        indxs_for_h = sorted_seq_lens_ind.unsqueeze(0).unsqueeze(-1).expand(hidden[0].shape[0], -1,
                                                                            hidden[0].shape[2])
        unsorted_h = (torch.zeros_like(hidden[0]), torch.zeros_like(hidden[1]))
        unsorted_h[0].scatter_(1, indxs_for_h, hidden[0])
        unsorted_h[1].scatter_(1, indxs_for_h, hidden[1])

        output = output.contiguous()
        max_output, _ = output.max(dim=1)

        return unsorted_output, unsorted_h, max_output

class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        hidden_reduced_h = F.relu(self.reduce_h(h.view(-1, config.hidden_dim * 2)))
        hidden_reduced_c = F.relu(self.reduce_c(c.view(-1, config.hidden_dim * 2)))

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) # h, c dim = 1 x b x hidden_dim

class SectionAttention(nn.Module):
    def __init__(self):
        super(SectionAttention, self).__init__()
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, h, coverage):
        b, t_k, n = list(h.size())
        encoder_feature = self.W_h(h)

        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous()

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim

        e = F.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B * t_k x 1
        scores = scores.squeeze(-1)  # B x t_k

        attn_dist_ = F.softmax(scores, dim=1)
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor
        attn_dist = attn_dist.view(-1, t_k)  # B x t_k

        return attn_dist


class WordAttention(nn.Module):
    def __init__(self):
        super(WordAttention, self).__init__()
        # attention
        self.W_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        if config.is_coverage:
            self.W_c = nn.Linear(1, config.hidden_dim * 2, bias=False)
        self.decode_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.v = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, s_t_hat, h, coverage, enc_padding_mask, beta):
        b, s, t_k, n = list(h.size())
        encoder_feature = self.W_h(h)

        dec_fea = self.decode_proj(s_t_hat) # B x 2*hidden_dim
        dec_fea_expanded = dec_fea.unsqueeze(1).unsqueeze(2).expand(b, s, t_k, n).contiguous()

        att_features = encoder_feature + dec_fea_expanded # B * t_k x 2*hidden_dim
        if config.is_coverage:
            coverage_input = coverage.unsqueeze(-1)
            coverage_feature = self.W_c(coverage_input)  # B * t_k x 2*hidden_dim
            att_features = att_features + coverage_feature

        e = F.tanh(att_features) # B * t_k x 2*hidden_dim
        scores = self.v(e)  # B x s x t_k x 1
        scores = scores.squeeze(-1)  # B x s X t_k

        # Use section weighting (beta)
        weighted_scores = beta.unsqueeze(-1) * scores

        # Flatten view for softmax on all words
        weighted_scores = weighted_scores.view(config.batch_size, weighted_scores.shape[1] * weighted_scores.shape[2])
        enc_padding_mask = enc_padding_mask.view(config.batch_size,
                                                 enc_padding_mask.shape[1] * enc_padding_mask.shape[2])


        attn_dist_ = F.softmax(weighted_scores, dim=1) * enc_padding_mask
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x t_k
        h = h.contiguous().view(-1, t_k * s, n)  # B x t_k x 2*hidden_dim
        c_t = torch.bmm(attn_dist, h)  # B x 1 x n
        c_t = c_t.view(-1, config.hidden_dim * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(b, -1)  # B x t_k * s

        if config.is_coverage:
            coverage = coverage.clone() + attn_dist.view(*coverage.shape)

        return c_t, attn_dist, coverage

class SectionEncoder(nn.Module):
    def __init__(self):
        super(SectionEncoder, self).__init__()

        self.lstm = nn.LSTM(config.hidden_dim * 2, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)

    def forward(self, input):
        hidden = input[0].view(config.batch_size, config.max_num_sections, input[0].shape[2])
        cell = input[1].view(config.batch_size, config.max_num_sections, input[1].shape[2])

        # TODO: Make sure we don't need just hidden
        h_c = torch.cat((hidden, cell), dim=2)
        output, hidden = self.lstm(h_c)

        return output, hidden


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.word_attention = WordAttention()
        self.section_attention = SectionAttention()
        # decoder
        self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        init_wt_normal(self.embedding.weight)

        self.x_context = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.emb_dim)

        self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)

        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.hidden_dim * 4 + config.emb_dim, 1)

        #p_vocab
        self.out1 = nn.Linear(config.hidden_dim * 3, config.hidden_dim)
        self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
        init_linear_wt(self.out2)

    def forward(self, y_t_1, s_t_1, encoder_output, section_output, enc_padding_mask,
                c_t_1, extra_zeros, enc_batch_extend_vocab, coverage):
        y_t_1_embd = self.embedding(y_t_1)
        x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))

        lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
                             c_decoder.view(-1, config.hidden_dim)), 1)  # B x 2*hidden_dim

        # TODO: Use section padding if we support more than 4 sections
        beta = self.section_attention(s_t_hat, section_output, coverage=coverage)
        encoder_output = encoder_output.view(config.batch_size, config.max_num_sections, *encoder_output.shape[1:])
        c_t, attn_dist, coverage = self.word_attention(s_t_hat, encoder_output,
                                                       enc_padding_mask=enc_padding_mask, coverage=coverage,
                                                       beta=beta)

        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1) # B x hidden_dim * 3
        output = self.out1(output) # B x hidden_dim

        #output = F.relu(output)

        output = self.out2(output) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            enc_batch_extend_vocab = enc_batch_extend_vocab.view(*attn_dist_.shape)
            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, p_gen, coverage

class Model(object):
    def __init__(self, model_file_path=None, is_eval=False):
        encoder = Encoder()
        section_encoder = SectionEncoder()
        decoder = Decoder()
        reduce_state = ReduceState()
        section_reduce_state = ReduceState()

        # shared the embedding between encoder and decoder
        decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            section_encoder = section_encoder.eval()
            decoder = decoder.eval()
            reduce_state = reduce_state.eval()
            section_reduce_state = section_reduce_state.eval()

        if use_cuda:
            encoder = encoder.cuda()
            section_encoder = section_encoder.cuda()
            decoder = decoder.cuda()
            reduce_state = reduce_state.cuda()
            section_reduce_state = section_reduce_state.cuda()

        self.encoder = encoder
        self.section_encoder = section_encoder
        self.decoder = decoder
        self.reduce_state = reduce_state
        self.section_reduce_state = section_reduce_state

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.section_encoder.load_state_dict(state['section_encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)
            self.reduce_state.load_state_dict(state['reduce_state_dict'])
            self.section_reduce_state.load_state_dict(state['section_reduce_state_dict'])
