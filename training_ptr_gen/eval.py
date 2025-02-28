from __future__ import unicode_literals, print_function, division

import os
import time
import sys

import tensorflow as tf
import torch

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab

from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch
from model import Model

import logging
import shutil

use_cuda = config.use_gpu and torch.cuda.is_available()

class Evaluate(object):
    def __init__(self, model_file_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.eval_data_path, self.vocab, mode='eval',
                               batch_size=config.batch_size, single_pass=False)
        # time.sleep(15)
        self.model_file_path = model_file_path
        # model_name = os.path.basename(model_file_path)

        self.eval_dir = os.path.join(config.log_root, 'eval')
        # eval_dir = os.path.join(config.log_root, 'eval_%s' % (model_name))
        if not os.path.exists(self.eval_dir):
            os.mkdir(self.eval_dir)
        self.summary_writer = tf.summary.FileWriter(self.eval_dir)

        # self.model = Model(model_file_path, is_eval=True)

    def eval_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage, \
            enc_graph_batch, enc_graph_mask_batch, enc_node_type_mask_batch = get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        graph_enc_outputs, ast_encoder_outputs, graph_encoder_feature, ast_encoder_feature, graph_dec_in_state, \
            ast_dec_in_state = self.model.encoder(enc_batch, enc_lens, enc_graph_batch, enc_graph_mask_batch, enc_node_type_mask_batch)
        s_t_1 = self.model.reduce_state(graph_dec_in_state, ast_dec_in_state)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1,attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        graph_enc_outputs, ast_encoder_outputs, graph_encoder_feature,
                                                        ast_encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)
        return loss.item()

    def save_model(self, model):
        # 将model复制到self.eval_dir路径下
        shutil.copy(model, os.path.join(self.eval_dir, "best_checkpoint"))


    def run_eval(self):
        running_avg_loss, iter = 0, 0
        start = time.time()
        batch = self.batcher.next_batch()
        best_loss = None

        # 遍历路径下所有的模型文件，重新加载self.model
        all_models = os.listdir(self.model_file_path)
        # 排序
        all_models.sort(key=lambda x: int(x.split('_')[1]))
        for model_name in all_models:
            print("Evaluating model %s" % model_name)
            self.model = None
            torch.cuda.empty_cache()
            self.model = Model(self.model_file_path + '/' + model_name, is_eval=True)

        # while batch is not None:
            loss = self.eval_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1

            if best_loss is None or running_avg_loss < best_loss:
                print("Found new best model with %.3f running_avg_loss at %d. Saving to %s" % (
                    running_avg_loss, int(model_name.split('_')[1]), self.eval_dir))

                self.save_model(self.model_file_path + '/' + model_name)
                best_loss = running_avg_loss

            if iter % 100 == 0:
                self.summary_writer.flush()
            print_interval = 5
            # if iter % print_interval == 0:
            print('steps %d, seconds for %d batch: %.2f , loss: %f' % (
            iter, print_interval, time.time() - start, running_avg_loss))
            start = time.time()
            batch = self.batcher.next_batch()


if __name__ == '__main__':
    model_filename = sys.argv[1]
    eval_processor = Evaluate(model_filename)
    eval_processor.run_eval()


