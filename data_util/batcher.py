#Most of this file is copied form https://github.com/abisee/pointer-generator/blob/master/batcher.py

from queue import Queue
import time
from random import shuffle
from threading import Thread

import numpy as np
import tensorflow as tf

from data_util import config
from data_util import data

import random
random.seed(1234)


class Example(object):

  def __init__(self, article_sections, abstract_sentences, vocab, article_sents, similarity_scores):
    # Get ids of special tokens
    start_decoding = vocab.word2id(data.START_DECODING)
    stop_decoding = vocab.word2id(data.STOP_DECODING)

    section_words = []
    self.enc_lens = []
    self.enc_inputs = []

    self.sent_lens = []

    # Process the article
    for i, section in enumerate(article_sections[:config.max_num_sections]):
      current_article_sents = [x.split() for x in article_sents[i]]
      words = section.split()
      if len(words) > config.max_section_size:
        words_to_remove = len(words) - config.max_section_size
        words = words[:config.max_section_size]

        indx = 0
        inner_stay = 0
        for j, sent in enumerate(reversed(current_article_sents)):
          indx += len(sent)
          if indx > words_to_remove:
            inner_stay = indx - words_to_remove
            break

        if j > 0:
          current_article_sents = current_article_sents[: -j]
        if inner_stay > 0:
          current_article_sents[-1] = current_article_sents[-1][: inner_stay]

      assert sum([len(x) for x in current_article_sents]) == len(words), "Bug in sent filtering!"

      self.enc_lens.append(len(words))
      self.enc_inputs.append([vocab.word2id(w) for w in words]) # list of word ids; OOVs are represented by the id for UNK token

      self.sent_lens.append([len(x) for x in current_article_sents])

      section_words.append(words)

    self.num_sections = len(section_words)
    self.max_enc_len = max(self.enc_lens)
    self.max_num_sents = max([len(x) for x in self.sent_lens])

    # Process the abstract
    abstract = ' '.join(abstract_sentences) # string
    abstract_words = abstract.split() # list of strings
    abs_ids = [vocab.word2id(w) for w in abstract_words] # list of word ids; OOVs are represented by the id for UNK token

    # Get the decoder input sequence and target sequence
    self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, config.max_dec_steps, start_decoding, stop_decoding)
    self.dec_len = len(self.dec_input)

    # If using pointer-generator mode, we need to store some extra info
    if config.pointer_gen:
        # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
      self.enc_inputs_extend_vocab, self.article_oovs = data.article2ids(section_words, vocab)

      # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
      abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

      # Overwrite decoder target sequence so it uses the temp article OOV ids
      _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, config.max_dec_steps, start_decoding, stop_decoding)

    # Process the similarity scores
    self.similarity_scores = []
    assert len(similarity_scores) == sum([len(x) for x in article_sents])
    global_sent_index = 0
    for i, section_sent_lens in enumerate(self.sent_lens):
      num_sents_cut = len(section_sent_lens)
      self.similarity_scores += similarity_scores[global_sent_index : global_sent_index + num_sents_cut]
      global_sent_index += len(article_sents[i])

    # Store the original strings
    self.original_article = article_sections
    self.original_abstract = abstract
    self.original_abstract_sents = abstract_sentences
    self.original_article_sents = article_sents


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

  def pad_seq(self, seq, max_len, pad_id):
    return seq + [pad_id] * (max_len - len(seq))

  def pad_encoder_inputs(self, max_len, pad_id):
    for i in range(self.num_sections):
      self.enc_inputs[i] = self.pad_seq(self.enc_inputs[i], max_len, pad_id)
      if config.pointer_gen:
        self.enc_inputs_extend_vocab[i] = self.pad_seq(self.enc_inputs_extend_vocab[i], max_len, pad_id)


class Batch(object):
  def __init__(self, example_list, vocab, batch_size):
    self.batch_size = batch_size
    self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.init_encoder_seq(example_list) # initialize the input to the encoder
    self.init_decoder_seq(example_list) # initialize the input and targets for the decoder
    self.store_orig_strings(example_list) # store the original strings


  def init_encoder_seq(self, example_list):
    # Determine the maximum length of the encoder input sequence in this batch
    max_enc_seq_len = max([ex.max_enc_len for ex in example_list])
    max_num_sents = max([ex.max_num_sents for ex in example_list])
    max_sim_score_len = max([len(ex.similarity_scores) for ex in example_list])

    # Pad the encoder input sequences up to the length of the longest sequence
    for ex in example_list:
      ex.pad_encoder_inputs(max_enc_seq_len, self.pad_id)

    # Initialize the numpy arrays
    self.enc_batch = np.zeros((self.batch_size, config.max_num_sections, max_enc_seq_len), dtype=np.int32)
    self.enc_lens = np.zeros((self.batch_size, config.max_num_sections), dtype=np.int32)
    self.enc_padding_mask = np.zeros((self.batch_size, config.max_num_sections, max_enc_seq_len), dtype=np.float32)
    self.sent_lens = np.zeros((self.batch_size, config.max_num_sections, max_num_sents), dtype=np.int32)
    self.sim_scores = np.zeros((self.batch_size, max_sim_score_len), dtype=np.int32)
    self.sim_scores_mask = np.zeros((self.batch_size, max_sim_score_len), dtype=np.float32)

    # Fill in the numpy arrays
    assert min([len(x.enc_inputs) for x in example_list]) == 4, "Invalid num of sections!"

    for i, ex in enumerate(example_list):
      for j in range(len(ex.enc_inputs)):
        self.enc_batch[i, j, :len(ex.enc_inputs[j])] = ex.enc_inputs[j][:]
        self.enc_lens[i, j] = ex.enc_lens[j]
        self.enc_padding_mask[i, j, :ex.enc_lens[j]] = np.ones(ex.enc_lens[j])
        self.sent_lens[i, j, :len(ex.sent_lens[j])] = ex.sent_lens[j][:]
      self.sim_scores[i, :len(ex.similarity_scores)] = ex.similarity_scores[:]
      self.sim_scores_mask[i, :len(ex.similarity_scores)] = np.ones(len(ex.similarity_scores))

    # For pointer-generator mode, need to store some extra info
    if config.pointer_gen:
      # Determine the max number of in-article OOVs in this batch
      self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
      # Store the in-article OOVs themselves
      self.art_oovs = [ex.article_oovs for ex in example_list]
      # Store the version of the enc_batch that uses the article OOV ids
      self.enc_batch_extend_vocab = np.zeros((self.batch_size, config.max_num_sections, max_enc_seq_len), dtype=np.int32)
      for i, ex in enumerate(example_list):
        for j in range(ex.num_sections):
          self.enc_batch_extend_vocab[i, j, :len(ex.enc_inputs_extend_vocab[j])] = ex.enc_inputs_extend_vocab[j][:]

  def init_decoder_seq(self, example_list):
    # Pad the inputs and targets
    for ex in example_list:
      ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

    # Initialize the numpy arrays.
    self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
    self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
    self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
    self.dec_lens = np.zeros(self.batch_size, dtype=np.int32)

    # Fill in the numpy arrays
    for i, ex in enumerate(example_list):
      self.dec_batch[i, :] = ex.dec_input[:]
      self.target_batch[i, :] = ex.target[:]
      self.dec_lens[i] = ex.dec_len
      self.dec_padding_mask[i][:ex.dec_len] = np.ones(ex.dec_len)

  def store_orig_strings(self, example_list):
    self.original_articles = [ex.original_article for ex in example_list] # list of lists
    self.original_abstracts = [ex.original_abstract for ex in example_list] # list of lists
    self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list] # list of list of lists
    self.original_article_sents = [ex.original_article_sents for ex in example_list]  # list of list of lists



class Batcher(object):
  BATCH_QUEUE_MAX = 100 # max number of batches the batch_queue can hold

  def __init__(self, data_path, vocab, mode, batch_size, single_pass):
    self._data_path = data_path
    self._vocab = vocab
    self._single_pass = single_pass
    self.mode = mode
    self.batch_size = batch_size
    # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
    self._batch_queue = Queue(self.BATCH_QUEUE_MAX)
    self._example_queue = Queue(self.BATCH_QUEUE_MAX * self.batch_size)

    # Different settings depending on whether we're in single_pass mode or not
    if single_pass:
      self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
      self._num_batch_q_threads = 1  # just one thread to batch examples
      self._bucketing_cache_size = 1 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
      self._finished_reading = False # this will tell us when we're finished reading the dataset
    else:
      self._num_example_q_threads = 1 #16 # num threads to fill example queue
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
        # read the next example from file. article and abstract are both strings.
        (sections, abstract, article_sents, similarity_scores) = next(input_gen)
      except StopIteration: # if there are no more examples:
        tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
        if self._single_pass:
          tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
          self._finished_reading = True
          break
        else:
          raise Exception("single_pass mode is off but the example generator is out of data; error.")

      abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)] # Use the <s> and </s> tags in abstract to get a list of sentences.
      example = Example(sections, abstract_sentences, self._vocab, article_sents, similarity_scores) # Process into an Example.
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
        inputs = sorted(inputs, key=lambda inp: inp.max_enc_len, reverse=True) # sort by length of encoder sequence

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
      e = next(example_generator) # e is a tf.Example
      yield (e['article_sections'], e['abstract_text'], e['article_sents'], e['similarity_scores'])
