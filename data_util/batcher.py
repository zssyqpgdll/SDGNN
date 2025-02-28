#Most of this file is copied form https://github.com/abisee/pointer-generator/blob/master/batcher.py

import queue as Queue
import time
from random import shuffle
from threading import Thread

import numpy as np
import tensorflow as tf

from data_util import config
from data_util import data

import os
import json
import random
random.seed(1234)

# 加载graph的边类型字典
with open('./edge_vocab.json', 'r') as f:
    edge_vocab_dict = json.load(f)

# 加载node的节点类型字典
with open('./node_type.json', 'r') as f:
    node_type_vocab_dict = json.load(f)


class Example(object):

  def __init__(self, article, graph, abstract_sentences, node_type, layer_position, vocab):
    # Get ids of special tokens
    start_decoding = vocab.word2id(data.START_DECODING)
    stop_decoding = vocab.word2id(data.STOP_DECODING)

    # Process the article
    article_words = article.split('-+|.-+.|')
    if len(article_words) > config.max_enc_steps:
      article_words = article_words[:config.max_enc_steps]
    self.enc_len = len(article_words) # store the length after truncation but before padding
    self.enc_input = [vocab.word2id(w) for w in article_words] # list of word ids; OOVs are represented by the id for UNK token

    self.enc_graph = self.process_graph_data(graph)

    self.enc_node_type = [int(node_type_vocab_dict[type]) for type in node_type.split('-+|.-+.|')]
    if len(self.enc_node_type) > config.max_enc_steps:
      self.enc_node_type = self.enc_node_type[:config.max_enc_steps]
    self.enc_node_type_len = len(self.enc_node_type)

    if self.enc_len != self.enc_node_type_len:
      for i in range(10):
        print('self.enc_len != self.enc_node_type_len')
      os._exit()

    node_layer_position = layer_position.split('-+|.-+.|')
    if len(node_layer_position) > config.max_enc_steps:
      node_layer_position = node_layer_position[:config.max_enc_steps]
    self.enc_layer_position = [int(layer) for layer in node_layer_position]

    # Process the abstract
    abstract = '-+|.-+.|'.join(abstract_sentences) # string
    abstract_words = abstract.split('-+|.-+.|') # list of strings
    abstract_words = abstract_words[1:-1]
    abs_ids = [vocab.word2id(w) for w in abstract_words] # list of word ids; OOVs are represented by the id for UNK token


    # Get the decoder input sequence and target sequence
    self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding)
    self.dec_len = len(self.dec_input)

    # If using pointer-generator mode, we need to store some extra info
    if config.pointer_gen:
      # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
      self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)

      # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
      abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

      # Overwrite decoder target sequence so it uses the temp article OOV ids
      _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)

    # Store the original strings
    # self.original_article = article
    # self.original_abstract = abstract
    # self.original_abstract_sents = abstract_sentences
    # modify
    self.original_article = ' '.join(article.split('-+|.-+.|')).strip()
    self.original_abstract = ' '.join(abstract.split('-+|.-+.|')).strip()
    new_abstract_sentences = []
    for abstract_sentence in abstract_sentences:
      abstract_sentence = ' '.join(abstract_sentence.split('-+|.-+.|')).strip()
      new_abstract_sentences.append(abstract_sentence)

    self.original_abstract_sents = new_abstract_sentences


  def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len: # truncate
      inp = inp[:max_len]
      target = target[:max_len] # no end_token
    else: # no truncation
      target.append(stop_id) # end token
    assert len(inp) == len(target)
    return inp, target


  def pad_decoder_inp_targ(self, max_len, pad_id):
    while len(self.dec_input) < max_len:
      self.dec_input.append(pad_id)
    while len(self.target) < max_len:
      self.target.append(pad_id)


  def pad_encoder_input(self, max_len, pad_id):
    while len(self.enc_input) < max_len:
      self.enc_input.append(pad_id)
    if config.pointer_gen:
      while len(self.enc_input_extend_vocab) < max_len:
        self.enc_input_extend_vocab.append(pad_id)

  def pad_encoder_graph_adj(self, max_len):
    max_node_count = max_len # 这个批次内最大的节点数量
    num_edge_types = len(edge_vocab_dict) # 边类型种类
    adj_mat = np.zeros((2 * num_edge_types, max_node_count, max_node_count), dtype=np.int32) # 邻接矩阵
    # ast_adj_mat = np.zeros((2, max_node_count, max_node_count), dtype=np.int32) # 邻接矩阵

    for edge in self.enc_graph:
      src = edge[0]
      dest = edge[1]
      e = edge[2]
      adj_mat[e - 1, src, dest] = 1
      # 反向认为是另外一种边
      adj_mat[(e - 1) + num_edge_types, dest, src] = 1
      # if e == 1:
      #   ast_adj_mat[e - 1, src, dest] = 1
      #   ast_adj_mat[e, dest, src] = 1

    # 加上自连接(自环)
    adj_mat = adj_mat + np.eye(adj_mat.shape[1])
    # ast_adj_mat = ast_adj_mat + np.eye(ast_adj_mat.shape[1])
    self.enc_graph_adj = adj_mat
    # self.enc_ast_adj = ast_adj_mat

    adj_mask = np.zeros((config.max_layer + 1, max_node_count, max_node_count))
    adj_mask = adj_mask + np.eye(adj_mask.shape[1])
    for i in range(0, config.max_layer + 1):
      correspond_node_list_cur = [index for (index, layer_position) in enumerate(self.enc_layer_position) if
                                  (layer_position == i)]
      correspond_node_list_next = [index for (index, layer_position) in enumerate(self.enc_layer_position) if
                                   (layer_position == i + 1)]

      for node_i in correspond_node_list_cur:
        for node_j in correspond_node_list_next:
          adj_mask[i, node_i, node_j] = 1

      for node_i in correspond_node_list_next:
        for node_j in correspond_node_list_cur:
          adj_mask[i + 1, node_i, node_j] = 1
    self.enc_graph_adj_mask = adj_mask

  def pad_encoder_node_type_mask(self, max_len):
    max_node_count = max_len # 这个批次内最大的节点数量
    num_node_types = len(node_type_vocab_dict) # 节点类型种类
    node_type_mask = np.zeros((num_node_types, max_node_count, 1), dtype=np.int32) # mask矩阵

    for index, node_type in enumerate(self.enc_node_type):
      node_type_mask[node_type-1, index, 0] = 1

    self.enc_node_type_mask = node_type_mask

  def process_graph_data(self, one_graph):
    # (0,0,s) (0,1,d) (1,0,r) (1,1,s) (1,2,d) (2,1,r) (2,2,s)
    # ['(0,0,s)', '(0,1,d)', '(1,0,r)', '(1,1,s)', '(1,2,d)', '(2,1,r)', '(2,2,s)']
    one_graph = one_graph.split(' ')
    one_graph_list = []

    if (len(one_graph) == 1) and one_graph[0] == '': # 处理图数据为空的时候，因为只有一个节点，又没有自环和反向边，所以图没有数据
      pass
    else:
      for connect in one_graph:
        e = (connect.replace('(', '').replace(')', '').strip()).split(',')

        src = int(e[0])
        dest = int(e[1])
        edge = int(edge_vocab_dict[e[2]])

        if src >= config.max_enc_steps or dest >= config.max_enc_steps:
          continue
        one_graph_list.append([src, dest, edge])

    return one_graph_list


class Batch(object):
  def __init__(self, example_list, vocab, batch_size):
    self.batch_size = batch_size
    self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.init_encoder_seq(example_list) # initialize the input to the encoder
    self.init_encoder_graph(example_list)  # 初始化邻接矩阵数据
    self.init_encoder_node_type(example_list)  # 初始化节点类型mask矩阵
    self.init_decoder_seq(example_list) # initialize the input and targets for the decoder
    self.store_orig_strings(example_list) # store the original strings


  def init_encoder_seq(self, example_list):
    # Determine the maximum length of the encoder input sequence in this batch
    max_enc_seq_len = max([ex.enc_len for ex in example_list])

    # Pad the encoder input sequences up to the length of the longest sequence
    for ex in example_list:
      ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

    # Initialize the numpy arrays
    # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
    self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
    self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
    self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.enc_batch[i, :] = ex.enc_input[:]
      self.enc_lens[i] = ex.enc_len
      for j in range(ex.enc_len):
        self.enc_padding_mask[i][j] = 1

    # For pointer-generator mode, need to store some extra info
    if config.pointer_gen:
      # Determine the max number of in-article OOVs in this batch
      self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
      # Store the in-article OOVs themselves
      self.art_oovs = [ex.article_oovs for ex in example_list]
      # Store the version of the enc_batch that uses the article OOV ids
      self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
      for i, ex in enumerate(example_list):
        self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

  def init_decoder_seq(self, example_list):
    # Pad the inputs and targets
    for ex in example_list:
      ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

    # Initialize the numpy arrays.
    self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
    self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
    self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
    self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.dec_batch[i, :] = ex.dec_input[:]
      self.target_batch[i, :] = ex.target[:]
      self.dec_lens[i] = ex.dec_len
      for j in range(ex.dec_len):
        self.dec_padding_mask[i][j] = 1

  def store_orig_strings(self, example_list):
    self.original_articles = [ex.original_article for ex in example_list] # list of lists
    self.original_abstracts = [ex.original_abstract for ex in example_list] # list of lists
    self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list] # list of list of lists

  def init_encoder_graph(self, example_list):
    max_enc_seq_len = max([ex.enc_len for ex in example_list])
    for ex in example_list:
      ex.pad_encoder_graph_adj(max_enc_seq_len)

    num_edge_types = len(edge_vocab_dict)  # 边类型种类
    self.enc_graph_batch = np.zeros((len(example_list), 2 * num_edge_types, max_enc_seq_len, max_enc_seq_len), dtype=np.int32)
    # self.enc_ast_batch = np.zeros((len(example_list), 2, max_enc_seq_len, max_enc_seq_len), dtype=np.int32)
    self.enc_graph_mask_batch = np.zeros((len(example_list), config.max_layer + 1, max_enc_seq_len, max_enc_seq_len), dtype=np.int32)
    for i, ex in enumerate(example_list):
      self.enc_graph_batch[i] = ex.enc_graph_adj
      # self.enc_ast_batch[i] = ex.enc_ast_adj
      self.enc_graph_mask_batch[i] = ex.enc_graph_adj_mask

  def init_encoder_node_type(self, example_list):
    max_enc_seq_len = max([ex.enc_len for ex in example_list])
    for ex in example_list:
      ex.pad_encoder_node_type_mask(max_enc_seq_len)

    num_node_types = len(node_type_vocab_dict)  # 节点类型种类
    self.enc_node_type_mask_batch = np.zeros((len(example_list), num_node_types, max_enc_seq_len, 1), dtype=np.int32)
    for i, ex in enumerate(example_list):
      self.enc_node_type_mask_batch[i] = ex.enc_node_type_mask


class Batcher(object):
  BATCH_QUEUE_MAX = 100 # max number of batches the batch_queue can hold

  def __init__(self, data_path, vocab, mode, batch_size, single_pass):
    self._data_path = data_path
    self._vocab = vocab
    self._single_pass = single_pass
    self.mode = mode
    self.batch_size = batch_size
    # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
    self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
    self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)

    # Different settings depending on whether we're in single_pass mode or not
    if single_pass:
      self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
      self._num_batch_q_threads = 1  # just one thread to batch examples
      self._bucketing_cache_size = 1 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
      self._finished_reading = False # this will tell us when we're finished reading the dataset
    else:
      self._num_example_q_threads = 8 #16 # num threads to fill example queue
      self._num_batch_q_threads = 1 #4  # num threads to fill batch queue
      self._bucketing_cache_size = 1 #100 # how many batches-worth of examples to load into cache before bucketing

    # Start the threads that load the queues
    self._example_q_threads = []
    for _ in range(self._num_example_q_threads):
      self._example_q_threads.append(Thread(target=self.fill_example_queue))
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()
    self._batch_q_threads = []
    for _ in range(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()

    # Start a thread that watches the other threads and restarts them if they're dead
    if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
      self._watch_thread = Thread(target=self.watch_threads)
      self._watch_thread.daemon = True
      self._watch_thread.start()

  def next_batch(self):
    # If the batch queue is empty, print a warning
    if self._batch_queue.qsize() == 0:
      tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
      if self._single_pass and self._finished_reading:
        tf.logging.info("Finished reading dataset in single_pass mode.")
        return None

    batch = self._batch_queue.get() # get the next Batch
    return batch

  def fill_example_queue(self):
    input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))

    while True:
      try:
        (article, graph, abstract, node_type, layer_position) = next(input_gen) # read the next example from file. article and abstract are both strings.
      except StopIteration: # if there are no more examples:
        tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
        if self._single_pass:
          tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
          self._finished_reading = True
          break
        else:
          raise Exception("single_pass mode is off but the example generator is out of data; error.")

      abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)] # Use the <s> and </s> tags in abstract to get a list of sentences.
      example = Example(article, graph, abstract_sentences, node_type, layer_position, self._vocab) # Process into an Example.
      self._example_queue.put(example) # place the Example in the example queue.

  def fill_batch_queue(self):
    while True:
      if self.mode == 'decode':
        # beam search decode mode single example repeated in the batch
        ex = self._example_queue.get()
        b = [ex for _ in range(self.batch_size)]
        self._batch_queue.put(Batch(b, self._vocab, self.batch_size))
      else:
        # Get bucketing_cache_size-many batches of Examples into a list, then sort
        inputs = []
        for _ in range(self.batch_size * self._bucketing_cache_size):
          inputs.append(self._example_queue.get())
        inputs = sorted(inputs, key=lambda inp: inp.enc_len, reverse=True) # sort by length of encoder sequence

        # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
        batches = []
        for i in range(0, len(inputs), self.batch_size):
          batches.append(inputs[i:i + self.batch_size])
        if not self._single_pass:
          shuffle(batches)
        for b in batches:  # each b is a list of Example objects
          self._batch_queue.put(Batch(b, self._vocab, self.batch_size))

  def watch_threads(self):
    while True:
      tf.logging.info(
        'Bucket queue size: %i, Input queue size: %i',
        self._batch_queue.qsize(), self._example_queue.qsize())

      tf.logging.info("time.sleep(60)...")
      time.sleep(60)
      for idx,t in enumerate(self._example_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found example queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_example_queue)
          self._example_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self._batch_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_batch_queue)
          self._batch_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()


  def text_generator(self, example_generator):
    while True:
      e = next(example_generator)  # e is a tf.Example
      try:
        src_text = e.features.feature['article'].bytes_list.value[
          0].decode()  # the article text was saved under the key 'article' in the data files
        article_text = e.features.feature['node'].bytes_list.value[0].decode()
        graph = e.features.feature['graph'].bytes_list.value[
          0].decode()  # the graph was saved under the key 'graph' in the data files
        abstract_text = e.features.feature['abstract'].bytes_list.value[
          0].decode()  # the abstract text was saved under the key 'abstract' in the data files
        node_type = e.features.feature['node_type'].bytes_list.value[
          0].decode()  # the abstract text was saved under the key 'abstract' in the data files
        layer_position = e.features.feature['layer_position'].bytes_list.value[
          0].decode()  # the abstract text was saved under the key 'abstract' in the data files

      except ValueError:
        tf.logging.error('Failed to get article or abstract from example')
        continue
      if len(article_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
        #tf.logging.warning('Found an example with empty article text. Skipping it.')
        continue
      else:
        yield (article_text, graph, abstract_text, node_type, layer_position)
