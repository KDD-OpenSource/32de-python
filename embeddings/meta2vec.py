from numbers import Number
from typing import Tuple, List
import tensorflow as tf
import numpy as np
import random


class Input:
    @classmethod
    def from_json(cls, data):
        raise NotImplementedError()

    def input(self) -> Tuple:
        raise NotImplementedError()


class MetaPathsInput(Input):

    def __init__(self,
                 meta_paths: List[List[Number]],
                 node_types: List[Number],
                 windows_size: Number = 2,
                 padding_value: Number = 0,
                 random_seed: Number = 42):
        self.meta_paths = meta_paths
        self.nodes = []
        self.node_types = node_types
        self.random_seed = random_seed

        self.window_size = windows_size
        self.padding_value = padding_value

    def set_window_size(self, window_size: Number):
        """
        Set the window size. This number of nodes are each taken from the left and right of the feature node
        for the context.
        :param window_size: the value it should be set to.
        :return: the object itself.
        """
        self.window_size = window_size
        return self

    def set_padding_value(self, padding_value: Number):
        """
        Set the padding value. With this value meta-paths will be padded at the beginning and end. This value will,
        for example, occur when the first or last node context is extracted.
        :param padding_value: the value it should be set to.
        :return: the object itself.
        """
        self.padding_value = padding_value
        return self

    @classmethod
    def from_json(cls, json, seperator = " | "):
        meta_paths = []
        node_types = set()
        for meta_paths in json.keys():
            node_ids = [int(id) for id in meta_paths.split(seperator)]
            meta_paths.append(node_ids)
            node_types |= set(node_ids)

        return cls(meta_paths, node_types)

    def _apply_transformation(self, meta_paths):
        features = {'node': [], 'context': []}
        for paths in meta_paths:
            for node_key in range(len(paths)):
                node = paths[node_key]
                left_keys = self._left_context(node_key, self.window_size)
                right_keys = self._right_context(node_key + 1, len(paths), self.window_size)
                context_keys = np.array(left_keys + right_keys) + 1
                context = np.array([self.padding_value - 1] + paths, dtype = np.float32) + 1
                context = context[context_keys]

                features['node'].append(node)
                features['context'].append(context)
        # Finally convert to array
        features['context'] = np.array(features['context'])
        return features['node'], features['context']

    @staticmethod
    def _left_context(max_key, win_size):
        return MetaPathsInput._primitive_context(list(range(max_key)), win_size)

    @staticmethod
    def _right_context(min_key, max_key, win_size):
        return MetaPathsInput._primitive_context(list(range(min_key, max_key)), win_size)

    @staticmethod
    def _primitive_context(key_range, windows_size):
        if len(key_range) < windows_size:
            return [-1] * (windows_size - len(key_range)) + key_range
        return random.sample(key_range, windows_size)

    def _update(self):
        if len(self.nodes) < 1:
            self.nodes, self.contexts = self._apply_transformation(self.meta_paths)

    def skip_gram_input(self) -> tf.data.Dataset:
        """
        Get the dataset to train on in skip-gram format.
        :return: the dataset with nodes as features and context as labels.
        """
        self._update()
        return tf.data.Dataset().from_tensor_slices(({'node': self.nodes}, self.contexts))

    def bag_of_words_input(self) -> tf.data.Dataset:
        """
        Get the dataset to train on in continuous bag of words format.
        :return: the dataset with context as features and nodes as labels.
        """
        self._update()
        return tf.data.Dataset().from_tensor_slices(({'context': self.contexts}, self.nodes))

    def get_vocab_size(self) -> Number:
        return len(self.nodes)

    def get_vocab(self) -> List[Number]:
        return self.nodes


class NodeInput(Input):
    pass


def model_word2vec(features, labels, mode, params):
    """
    Word2vec model from "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al.)
    :return:
    """
    input = tf.feature_column.input_layer(features, params['feature_columns'])

    size_of_vocabulary = input.shape[1]

    word_embeddings = tf.Variable(
        initial_value=tf.random_uniform(shape=[size_of_vocabulary, params['embedding_size']], minval=-1, maxval=1),
        name='word_embeddings')

    # Look up embedding for all words
    embedded_words = tf.nn.embedding_lookup(word_embeddings, tf.argmax(input, axis=0))

    return _model_word2vec(mode, size_of_vocabulary, params['loss'], labels, embedded_words)


def _model_word2vec(mode, size_of_vocabulary, loss, labels, embedded_words):
    # Concatenate vectors
    concatenated_embeddings = embedded_words

    # Transform embeddings linearly
    hidden_layer = tf.layers.dense(inputs=concatenated_embeddings, units=size_of_vocabulary, activation=None,
                                   use_bias=True,
                                   name="linear_transformation",
                                   kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

    # Apply softmax and calulate loss
    if loss == 'cross_entropy':
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_layer, labels=labels),
                              name="cross_entropy_loss")
    else:
        raise NotImplementedError("This loss isn't implemented yet. Implement it!")

    optimizer = tf.train.AdamOptimizer()
    total_train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    metrics = {'loss': loss}
    # Loss for tensorboard
    tf.summary.scalar('Loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=total_train_op)
    else:
        raise NotImplementedError("This mode isn't implemented yet")


def model_paragraph_vectors_skipgram():
    """
    Distributed Memory Model of Paragraph Vectors (PV-DM) from "Distributed Representations of Sentences and Documents" (Mikolov et al.)
    :return:
    """
    pass


def model_paragraph_vectors_dbow():
    """
    Distributed Bag of Words version of Paragraph Vector (PV-DBOW) from "Distributed Representations of Sentences and Documents" (Mikolov et al.)
    :return:
    """
    pass


def create_estimator():
    pass


def parse_arguments():
    pass


if __name__ == "__main__":
    args = parse_arguments()
    classifier = create_estimator()

    if args.mode == 'train':
        # TODO: Create Dataset
        dataset = tf.data.Dataset()
        classifier.train(input_fn=MetaPathsInput(dataset).input())
    elif args.mode == 'predict':
        raise NotImplementedError()
    elif args.mode == 'eval':
        raise NotImplementedError()
