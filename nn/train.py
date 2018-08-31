from __future__ import unicode_literals, print_function, division

import os
import time
import sys

import tensorflow as tf
import torch
from nn.model import Model
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from nn.custom_adagrad import AdagradCustom

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from nn.train_util import get_input_from_batch, get_output_from_batch

use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(15)

        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.summary_writer = tf.summary.FileWriter(train_dir)

    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'section_encoder_state_dict': self.model.section_encoder.state_dict(),
            'sentence_filterer_state_dict': self.model.sentence_filterer.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'section_reduce_state_dict': self.model.section_reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + \
                 list(self.model.section_encoder.parameters()) + \
                 list(self.model.sentence_filterer.parameters()) + \
                 list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters()) + \
                 list(self.model.section_reduce_state.parameters())
        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = AdagradCustom(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage and not config.is_sentence_filtering:
                self.optimizer.load_state_dict(state['optimizer'])
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()

        return start_iter, start_loss

    def train_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, sent_lens = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()

        encoder_outputs, encoder_hidden, max_encoder_output = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)
        if config.use_maxpool_init_ctx:
            c_t_1 = max_encoder_output

        gamma = None
        if config.is_sentence_filtering:
            gamma, sent_dists = self.model.sentence_filterer(encoder_outputs, sent_lens)

        section_outputs, section_hidden = self.model.section_encoder(s_t_1)
        s_t_1 = self.model.section_reduce_state(section_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1, attn_dist, p_gen, coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, section_outputs, enc_padding_mask,
                                                        c_t_1, extra_zeros, enc_batch_extend_vocab,
                                                        coverage, gamma)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage.view(*attn_dist.shape)), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        if config.is_sentence_filtering:
            sim_scores = torch.FloatTensor(batch.sim_scores)
            if use_cuda:
                sim_scores = sim_scores.cuda()
            sent_filter_loss = F.binary_cross_entropy(sent_dists, sim_scores)
            loss += config.sent_loss_wt * sent_filter_loss

        loss.backward()

        clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.section_encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.sentence_filterer.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.section_reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        while iter < n_iters:
            print('\rBatch %d' % iter, end="")
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if iter % 5000 == 0:
                self.summary_writer.flush()
            print_interval = 10
            if iter % print_interval == 0:
                print(' steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                            time.time() - start, loss))
                start = time.time()
            if iter % 1000 == 0:
                self.save_model(running_avg_loss, iter)

if __name__ == '__main__':
    train_processor = Train()
    if len(sys.argv) > 1:
        model_filename = sys.argv[1]
    else:
        model_filename = None

    train_processor.trainIters(config.max_iterations, model_filename)
