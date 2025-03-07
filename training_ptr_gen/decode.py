#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

from __future__ import unicode_literals, print_function, division

import sys
import importlib,sys
importlib.reload(sys)

import os
import time
import tensorflow as tf
import torch
from torch.autograd import Variable

from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util import data, config
from model import Model
from data_util.utils import write_for_rouge, rouge_eval, rouge_log, bleu_eval
from train_util import get_input_from_batch
from compute_sore import eval

import re
import tqdm

use_cuda = config.use_gpu and torch.cuda.is_available()

class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      context = context,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
        model_name = os.path.basename(model_file_path)
        self.vocab = Vocab(config.vocab_path, config.vocab_size)

        # 不是文件夹，则为单模型解码
        if not os.path.isdir(self.model_file_path):
            self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                                   batch_size=config.beam_size, single_pass=True)
            config.batch_size = config.beam_size
            self.model = Model(model_file_path, is_eval=True)
            self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (model_name))
            self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
            self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
            for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
                if not os.path.exists(p):
                    os.mkdir(p)
        else:
            pattern = r'train_\d+'
            match = re.search(pattern, os.path.normpath(model_file_path))
            if match:
                extracted_string = match.group(0)
            else:
                print("No match found")
                exit()
            train_dir = extracted_string
            self.all_decode_dir = os.path.join(config.log_root, train_dir, 'decode')
            if not os.path.exists(self.all_decode_dir):
                os.mkdir(self.all_decode_dir)

        config.batch_size = config.beam_size
        # time.sleep(15)


    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


    def decode(self):
        start = time.time()

        if not os.path.isdir(self.model_file_path):
            counter = 0
            batch = self.batcher.next_batch()
            while batch is not None:
                # Run beam search to get best Hypothesis
                best_summary = self.beam_search(batch)

                # Extract the output ids from the hypothesis and convert back to words
                output_ids = [int(t) for t in best_summary.tokens[1:]]
                decoded_words = data.outputids2words(output_ids, self.vocab,
                                                     (batch.art_oovs[0] if config.pointer_gen else None))

                # Remove the [STOP] token from decoded_words, if necessary
                try:
                    fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                    decoded_words = decoded_words[:fst_stop_idx]
                except ValueError:
                    decoded_words = decoded_words

                original_abstract_sents = batch.original_abstracts_sents[0]

                write_for_rouge(original_abstract_sents, decoded_words, counter,
                                self._rouge_ref_dir, self._rouge_dec_dir)
                counter += 1
                # if counter % 1000 == 0:
                print('example %d finished in %d sec' % (counter, time.time() - start))
                start = time.time()

                batch = self.batcher.next_batch()

            print("Decoder has finished reading dataset for single_pass.")
            print("Now starting ROUGE eval...")
            bleu, rouge_score = eval(self._rouge_ref_dir, self._rouge_dec_dir)

        else:
            best_bleu4, best_rougel = 0, 0
            best_bleu4_new, best_rougel_new = 0, 0
            best_bleu4_iter, best_rougel_iter = None, None
            best_bleu4_iter_new, best_rougel_iter_new = None, None

            # 遍历路径下所有的模型文件，重新加载self.model
            all_models = os.listdir(self.model_file_path)
            # 排序
            all_models.sort(key=lambda x: int(x.split('_')[1]))
            all_models = [model for model in all_models if
                          int(model.split('_')[1]) <= 2000 and int(model.split('_')[1]) > 1800]
            for model_name in tqdm.tqdm(all_models):
                #if int(model_name.split('_')[1]) <= 1960:
                #    continue
                counter = 0
                print("Evaluating model %s" % model_name)
                self._decode_dir = os.path.join(self.all_decode_dir, 'decode_%s' % (model_name))
                self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
                self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
                for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
                    if not os.path.exists(p):
                        os.mkdir(p)
                self.model = None
                torch.cuda.empty_cache()
                self.model = Model(self.model_file_path + '/' + model_name, is_eval=True)

                self.batcher, batch = None, None
                self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                                       batch_size=config.beam_size, single_pass=True)
                batch = self.batcher.next_batch()
                while batch is not None:
                    # Run beam search to get best Hypothesis
                    best_summary = self.beam_search(batch)

                    # Extract the output ids from the hypothesis and convert back to words
                    output_ids = [int(t) for t in best_summary.tokens[1:]]
                    decoded_words = data.outputids2words(output_ids, self.vocab,
                                                         (batch.art_oovs[0] if config.pointer_gen else None))

                    # Remove the [STOP] token from decoded_words, if necessary
                    try:
                        fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                        decoded_words = decoded_words[:fst_stop_idx]
                    except ValueError:
                        decoded_words = decoded_words

                    original_abstract_sents = batch.original_abstracts_sents[0]

                    write_for_rouge(original_abstract_sents, decoded_words, counter,
                                    self._rouge_ref_dir, self._rouge_dec_dir)
                    counter += 1
                    # if counter % 1000 == 0:
                    print('example %d finished in %d sec'%(counter, time.time() - start))
                    start = time.time()

                    batch = self.batcher.next_batch()

                print("Decoder has finished reading dataset for single_pass.")
                print("Now starting ROUGE eval...")
                bleu, rouge_score, bleu_new, rouge_score_new = eval(self._rouge_ref_dir, self._rouge_dec_dir)
                if best_bleu4 < bleu:
                    best_bleu4 = bleu
                    best_bleu4_iter = model_name
                if best_rougel < rouge_score:
                    best_rougel = rouge_score
                    best_rougel_iter = model_name
                if best_bleu4_new < bleu_new:
                    best_bleu4_new = bleu_new
                    best_bleu4_iter_new = model_name
                if best_rougel_new < rouge_score_new:
                    best_rougel_new = rouge_score_new
                    best_rougel_iter_new = model_name
                print("best bleu4: {}, model: {}".format(best_bleu4, best_bleu4_iter))
                print("best rougel: {}, model: {}".format(best_rougel, best_rougel_iter))
                print("best bleu4_new: {}, model: {}".format(best_bleu4_new, best_bleu4_iter_new))
                print("best rougel_new: {}, model: {}".format(best_rougel_new, best_rougel_iter_new))

            print("final best bleu4: {}, model: {}".format(best_bleu4, best_bleu4_iter))
            print("final best rougel: {}, model: {}".format(best_rougel, best_rougel_iter))

        print("Decode Done!")

        # results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        # results_dict['bleu'] = bleu_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        # rouge_log(results_dict, self._decode_dir)

    def beam_search(self, batch):
        #batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0, \
            enc_graph_batch, enc_graph_mask_batch, enc_node_type_mask_batch = get_input_from_batch(batch, use_cuda)

        graph_enc_outputs, ast_encoder_outputs, graph_encoder_feature, ast_encoder_feature, graph_dec_in_state, \
            ast_dec_in_state = self.model.encoder(enc_batch, enc_lens, enc_graph_batch, enc_graph_mask_batch, enc_node_type_mask_batch)
        s_t_0 = self.model.reduce_state(graph_dec_in_state, ast_dec_in_state)

        dec_h, dec_c = s_t_0 # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = c_t_0[0],
                      coverage=(coverage_t_0[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
                        graph_enc_outputs, ast_encoder_outputs, graph_encoder_feature, ast_encoder_feature,
                        enc_padding_mask, c_t_1, extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)
            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

if __name__ == '__main__':
    model_filename = sys.argv[1]
    beam_Search_processor = BeamSearch(model_filename)
    beam_Search_processor.decode()



