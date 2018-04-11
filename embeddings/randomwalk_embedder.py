import urllib.request
import collections
import math
import os
import random
import zipfile
from collections import deque
import datetime as dt

import numpy as np
import tensorflow as tf


class LongWalkBatchGenerator:

    def __init__(self, file_name, skip_window):
        self.walk_list, self.available_nodes = self.read_walks(file_name)
        self.skip_window = skip_window
        self.iterator = self.global_window_iterator(self.walk_list, self.skip_window)

    @staticmethod
    def global_window_iterator(self, walk_list, skip_window):
        span_size = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
        for walk in walk_list:
            prepared_walk_list = self.prepare_edge_for_window(walk, skip_window)
            for window in self.sliding_window(prepared_walk_list, span_size):
                yield window

    # generate batch data from long walks: the walk is read as a "sentence" using a sliding window
    def generate_batch(self, batch_size, num_skips):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * self.skip_window
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        batch_index = 0
        context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        for i in range(batch_size // num_skips):
            window  = next(self.iterator, None)

            # if iterator has finished restart it
            if window is None:
                self.iterator = self.global_window_iterator(self.walk_list, self.skip_window)
                window = next(self.iterator, None)

            window[0], window[len(window) / 2] = window[len(window) / 2], window[
                0]  # swap middle with first element
            focus_node = window[0]

            selected_context_nodes = random.sample(range(1, len(window)), num_skips)
            for selection in selected_context_nodes:
                if window[selection] is None:
                    continue
                batch[batch_index] = focus_node
                context[batch_index, 0] = window[selection]
                batch_index += 1

        return batch, context

    @staticmethod
    def sliding_window(self, seq, span_size):
        it = iter(seq)
        win = deque((next(it, None) for _ in range(span_size)), maxlen=span_size)
        yield win
        append = win.append
        for e in it:
            append(e)
            yield win

    @staticmethod
    def prepare_edge_for_window(self, seq, edge_size):
        dummy = [None] * edge_size
        prepared = dummy + seq + dummy
        return prepared

    @staticmethod
    def read_walks(self, file_name):
        walk_list = []
        available_nodes = set()
        with open(file_name) as file:
            for line in file:
                node_ids = [int(id) for id in line.split()]

                walk_list.append(node_ids)
                available_nodes |= set(node_ids)
        return walk_list, available_nodes


class ShortWalkBatchGenerator:

    def __init__(self, file_name):
        self.walk_list, self.available_nodes = self.read_walks(file_name=file_name)
        self.data_index = 0

    @staticmethod
    def read_walks(file_name):
        walk_list = []
        available_nodes = set()
        with open(file_name) as file:
            for line in file:
                node_ids = line.split()

                start_id = int(node_ids[0])
                walk_id_list = [int(id) for id in node_ids[1:]]

                walk_list.append([start_id, walk_id_list])
                available_nodes.add(start_id)
        return walk_list, available_nodes

    # generate batch data from short walks: the whole walk is taken as the context for the start node
    # TODO should nodes at the beginning of a walk have a higher selection probabilty?
    def generate_batch(self, batch_size, num_skips):
        assert batch_size % num_skips == 0
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        for i in range(batch_size // num_skips):
            current_walk = self.walk_list[self.data_index]
            focus_node = current_walk[0]
            walk = current_walk[1]

            assert len(current_walk) >= num_skips

            selected_context_nodes = random.sample(range(0, len(walk)), num_skips)
            for counter, selection in enumerate(selected_context_nodes):
                batch[i * num_skips + counter] = focus_node
                context[i * num_skips + counter, 0] = walk[selection]

                self.data_index = (self.data_index + 1) % len(self.walk_list)
        return batch, context


batch_generator = ShortWalkBatchGenerator("output.csv")
embedded_nodes_size = len(batch_generator.available_nodes)

batch_size = 128
embedding_vector_size = 300  # Dimension of the embedding vector.
num_skips = 2         # How many times to reuse a walk to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_examples = random.sample(batch_generator.available_nodes, valid_size)
num_sampled = 64    # Number of negative examples to sample. (relevant for NCE loss)

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_context = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Look up embeddings for inputs.
  embeddings = tf.Variable(
      tf.random_uniform([embedded_nodes_size, embedding_vector_size], -1.0, 1.0))
  embed = tf.nn.embedding_lookup(embeddings, train_inputs)

  # Construct the variables for the softmax
  weights = tf.Variable(
      tf.truncated_normal([embedding_vector_size, embedded_nodes_size],
                          stddev=1.0 / math.sqrt(embedding_vector_size)))
  biases = tf.Variable(tf.zeros([embedded_nodes_size]))
  hidden_out = tf.transpose(tf.matmul(tf.transpose(weights), tf.transpose(embed))) + biases

  # convert train_context to a one-hot format
  train_one_hot = tf.one_hot(train_context, embedded_nodes_size)

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=train_one_hot))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()


def run(graph, num_steps):
    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      init.run()
      print('Initialized')

      average_loss = 0
      for step in range(num_steps):
        batch_inputs, batch_context = batch_generator.generate_batch(batch_size, num_skips)
        print(batch_inputs)
        print(batch_context)
        feed_dict = {train_inputs: batch_inputs, train_context: batch_context}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, cross_entropy], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step ', step, ': ', average_loss)
          average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
          sim = similarity.eval()
          for index, valid_word in valid_examples:
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[index, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
              close_word = nearest[k]
              log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
      final_embeddings = normalized_embeddings.eval()

# num_steps = 100
# softmax_start_time = dt.datetime.now()
# run(graph, num_steps=num_steps)
# softmax_end_time = dt.datetime.now()
# print("Softmax method took {} minutes to run 100 iterations".format((softmax_end_time-softmax_start_time).total_seconds()))

with graph.as_default():

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([embedded_nodes_size, embedding_vector_size],
                            stddev=1.0 / math.sqrt(embedding_vector_size)))
    nce_biases = tf.Variable(tf.zeros([embedded_nodes_size]))

    nce_loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_context,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=embedded_nodes_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(nce_loss)

    # Add variable initializer.
    init = tf.global_variables_initializer()

num_steps = 50000
nce_start_time = dt.datetime.now()
run(graph, num_steps)
nce_end_time = dt.datetime.now()
print("NCE method took {} minutes to run 100 iterations".format((nce_end_time-nce_start_time).total_seconds()))
