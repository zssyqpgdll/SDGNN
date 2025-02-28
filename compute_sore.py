#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
@project: PyCharm
@file: compute_sore.py
@author: Shengqiang Zhang
@time: 2020/4/18 21:23
@mail: sqzhang77@gmail.com
"""

import sys
import os
import glob2
from nmt_bleu import compute_bleu
from nmt_rouge import rouge
import logging
import pyrouge
import glob
from data_util import bleu
from c2nl.inputters.timer import AverageMeter, Timer
from c2nl.eval.bleu import Bleu, nltk_corpus_bleu, corpus_bleu
from c2nl.eval.rouge import Rouge
from c2nl.eval.meteor import Meteor
from collections import OrderedDict, Counter

# python ../../../../compute_sore.py "reference/*.txt" "decoded/*.txt"


references_list = []
candidates_list = []

references_rouge_list = []
candidates_rouge_list = []

references_dict = {}
candidates_dict = {}

def normalize_answer(s):
    """Lower text and remove extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(lower(s))

def eval_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    precision, recall, f1 = 0, 0, 0
    if len(ground_truth) == 0:
        if len(prediction) == 0:
            precision, recall, f1 = 1, 1, 1
    else:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same != 0:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1

def compute_eval_score(prediction, ground_truths):
    assert isinstance(prediction, str)
    precision, recall, f1 = 0, 0, 0
    for gt in ground_truths:
        _prec, _rec, _f1 = eval_score(prediction, gt)
        if _f1 > f1:
            precision, recall, f1 = _prec, _rec, _f1
    return precision, recall, f1

def rouge_eval(ref_dir, dec_dir):
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def bleu_eval(ref_dir, dec_dir):
    ref_dir = ref_dir + '/'
    dec_dir = dec_dir + '/'
    ref = []
    dec = []
    for i, j in zip(sorted(glob.glob(dec_dir + '*.txt')), sorted(glob.glob(ref_dir + '*.txt'))):
        ref_tex = ''
        dec_tex = ''
        for k in open(i).readlines():
            dec_tex = dec_tex + k.strip()
        for l in open(j).readlines():
            ref_tex = ref_tex + l.strip()
        ref.append(ref_tex)
        dec.append(dec_tex)
    bleu_score = bleu.moses_multi_bleu(dec, ref)
    return bleu_score

def new_method(references_dict, candidates_dict):
    # scorers = {
    #     "Bleu": Bleu(),
    #     "Meteor": Meteor()
    # }
    results_dict = rouge_eval(sys.argv[1], sys.argv[2])
    results_dict['bleu'] = bleu_eval(sys.argv[1], sys.argv[2])
    # print(results_dict)
    from c2nl.eval.meteor.meteor import Meteor
    meteor = Meteor()
    results_dict['c2nl_meteor'] = meteor.compute_score(references_dict, candidates_dict)
    from pycocoevalcap.meteor.meteor import Meteor
    # meteor = Meteor()
    # print(f"references_dict: {references_dict}")
    # results_dict['pycocoevalcap_meteor'] = meteor.compute_score(candidates_dict, references_dict)
    print(results_dict)
    # scores = {}
    # for name, scorer in scorers.items():
    #     if name == 'Bleu':
    #         score, all_scores, bleu = scorer.compute_score(candidates_dict, references_dict, verbose=1)
    #         print(f"bleu score: {score}")
    #         print(f"bleu scores: {all_scores}")
    #         print(f"bleu: {bleu}")
    #     else:
    #         score, all_scores = scorer.compute_score(candidates_dict, references_dict)
    #     if isinstance(score, list):
    #         for i, sc in enumerate(score, 1):
    #             scores[name + str(i)] = sc
    #     else:
    #         scores[name] = score
    # print(scores)

def eval_accuracies(hypotheses, references):
    """An unofficial evalutation helper.
     Arguments:
        hypotheses: A mapping from instance id to predicted sequences.
        references: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    assert sorted(references.keys()) == sorted(hypotheses.keys())

    # Compute BLEU scores
    #bleu_scorer = Bleu(n=4)
    #_, _, bleu = bleu_scorer.compute_score(references, hypotheses, verbose=0)
    # print(f"bleu1: {bleu}")
    #bleu = compute_bleu(references, hypotheses, max_order=4)['bleu']
    #_, bleu, _ = nltk_corpus_bleu(hypotheses, references)
    _, bleu, ind_bleu = corpus_bleu(hypotheses, references)
    # print(f"bleu2: {bleu}")

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    # rouge_l, ind_rouge = rouge_calculator.compute_score(hypotheses, references)
    rouge_l, ind_rouge = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    meteor_calculator = Meteor()
    meteor, _ = meteor_calculator.compute_score(references, hypotheses)

    f1 = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()

    for key in references.keys():
        _prec, _rec, _f1 = compute_eval_score(hypotheses[key][0], references[key])
        precision.update(_prec)
        recall.update(_rec)
        f1.update(_f1)
    print(f"bleu4: {bleu * 100}")
    print(f"rouge_l: {rouge_l * 100}")
    print(f"meteor: {meteor * 100}")
    print(f"precision: {precision.avg * 100}")
    print(f"recall: {recall.avg * 100}")
    print(f"f1: {f1.avg * 100}")
    return bleu * 100, rouge_l * 100, meteor * 100, precision.avg * 100, \
           recall.avg * 100, f1.avg * 100


if __name__ == '__main__':
    references_files_path = glob2.glob(sys.argv[1] + '/*.txt')
    candidates_files_path = glob2.glob(sys.argv[2] + '/*.txt')

    for reference_file in references_files_path:
        with open(reference_file,'r') as f:
            reference = (f.read().strip()).split(' ')
            reference = [reference]
            references_list.append(reference)

    for candidate_file in candidates_files_path:
        with open(candidate_file,'r') as f:
            candidate = (f.read().strip()).split(' ')
            candidate = candidate
            candidates_list.append(candidate)

    i = 0
    for reference_rouge_file in references_files_path:
        with open(reference_rouge_file,'r') as f:
            reference = (f.read().strip())
            references_rouge_list.append(reference)
            references_dict[i] = [reference]
            i += 1

    i = 0
    for candidate_rouge_file in candidates_files_path:
        with open(candidate_rouge_file,'r') as f:
            candidate = (f.read().strip())
            candidates_rouge_list.append(candidate)
            candidates_dict[i] = [candidate]
            i += 1

    assert len(references_list) == len(candidates_list), 'must be euqal.'


    count = 0

    for index in range(len(references_list)):
        for word in candidates_list[index]:
            if '[UNK]' in word:
                count += 1

        # print('reference: {}'.format(references_list[index][0]))
        # print('candidate: {}'.format(candidates_list[index]))
        # print('\n')


    print('UNK count: {}'.format(count))
    print('\n')

    print('nmt corpus bleu4: {}'.format(compute_bleu(references_list, candidates_list, max_order=4)[0]))
    print('\n')




    rouge_score = rouge(candidates_rouge_list, references_rouge_list)
    print('nmt rouge_1/f_score: {}'.format(rouge_score['rouge_1/f_score']))
    print('nmt rouge_1/r_score: {}'.format(rouge_score['rouge_1/r_score']))
    print('nmt rouge_1/p_score: {}'.format(rouge_score['rouge_1/p_score']))
    print('\n')
    print('nmt rouge_2/f_score: {}'.format(rouge_score['rouge_2/f_score']))
    print('nmt rouge_2/r_score: {}'.format(rouge_score['rouge_2/r_score']))
    print('nmt rouge_2/p_score: {}'.format(rouge_score['rouge_2/p_score']))
    print('\n')
    print('nmt rouge_l/f_score: {}'.format(rouge_score['rouge_l/f_score']))
    print('nmt rouge_l/r_score: {}'.format(rouge_score['rouge_l/r_score']))
    print('nmt rouge_l/p_score: {}'.format(rouge_score['rouge_l/p_score']))

    # new_method(references_dict, candidates_dict)
    eval_accuracies(candidates_dict, references_dict)
    # new_method()


def eval(references_files_path, candidates_files_path):
    references_list = []
    candidates_list = []
    references_rouge_list = []
    candidates_rouge_list = []
    references_dict = {}
    candidates_dict = {}
    references_files_path = glob2.glob(references_files_path + '/*.txt')
    candidates_files_path = glob2.glob(candidates_files_path + '/*.txt')
    for reference_file in references_files_path:
        with open(reference_file,'r') as f:
            reference = (f.read().strip()).split(' ')
            reference = [reference]
            references_list.append(reference)

    for candidate_file in candidates_files_path:
        with open(candidate_file,'r') as f:
            candidate = (f.read().strip()).split(' ')
            candidate = candidate
            candidates_list.append(candidate)

    i = 0
    for reference_rouge_file in references_files_path:
        with open(reference_rouge_file,'r') as f:
            reference = (f.read().strip())
            references_rouge_list.append(reference)
            references_dict[i] = [reference]
            i += 1

    i = 0
    for candidate_rouge_file in candidates_files_path:
        with open(candidate_rouge_file,'r') as f:
            candidate = (f.read().strip())
            candidates_rouge_list.append(candidate)
            candidates_dict[i] = [candidate]
            i += 1

    assert len(references_list) == len(candidates_list), 'must be euqal.'
    count = 0

    for index in range(len(references_list)):
        for word in candidates_list[index]:
            if '[UNK]' in word:
                count += 1

    print('UNK count: {}'.format(count))
    print('\n')

    bleu4 = compute_bleu(references_list, candidates_list, max_order=4)[0]
    print('nmt corpus bleu4: {}'.format(bleu4))
    print('\n')

    rouge_score = rouge(candidates_rouge_list, references_rouge_list)
    print('nmt rouge_1/f_score: {}'.format(rouge_score['rouge_1/f_score']))
    print('nmt rouge_1/r_score: {}'.format(rouge_score['rouge_1/r_score']))
    print('nmt rouge_1/p_score: {}'.format(rouge_score['rouge_1/p_score']))
    print('\n')
    print('nmt rouge_2/f_score: {}'.format(rouge_score['rouge_2/f_score']))
    print('nmt rouge_2/r_score: {}'.format(rouge_score['rouge_2/r_score']))
    print('nmt rouge_2/p_score: {}'.format(rouge_score['rouge_2/p_score']))
    print('\n')
    print('nmt rouge_l/f_score: {}'.format(rouge_score['rouge_l/f_score']))
    print('nmt rouge_l/r_score: {}'.format(rouge_score['rouge_l/r_score']))
    print('nmt rouge_l/p_score: {}'.format(rouge_score['rouge_l/p_score']))

    #eval_accuracies(candidates_dict, references_dict)
    
    return bleu4, rouge_score['rouge_l/f_score']
