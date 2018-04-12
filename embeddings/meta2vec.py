from numbers import Number
from typing import Tuple, List
import tensorflow as tf
import numpy as np
import random
import argparse
import json


class Input:
    @classmethod
    def from_json(cls, data):
        raise NotImplementedError()

    def skip_gram_input(self) -> tf.data.Dataset:
        """
        Get the dataset to train on in skip-gram format.
        :return: the dataset with nodes as features and context as labels.
        """
        raise NotImplementedError()

    def bag_of_words_input(self) -> tf.data.Dataset:
        """
        Get the dataset to train on in continuous bag of words format.
        :return: the dataset with context as features and nodes as labels.
        """
        raise NotImplementedError()

    def get_vocab_size(self) -> Number:
        raise NotImplementedError()

    def get_vocab(self) -> List[Number]:
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

    def set_window_size(self, window_size: Number) -> 'MetaPathsInput':
        """
        Set the window size. This number of nodes are each taken from the left and right of the feature node
        for the context.
        :param window_size: the value it should be set to.
        :return: the object itself.
        """
        self.window_size = window_size
        return self

    def set_padding_value(self, padding_value: Number) -> 'MetaPathsInput':
        """
        Set the padding value. With this value meta-paths will be padded at the beginning and end. This value will,
        for example, occur when the first or last node context is extracted.
        :param padding_value: the value it should be set to.
        :return: the object itself.
        """
        self.padding_value = padding_value
        return self

    @classmethod
    def from_json(cls, json, seperator=" | ") -> 'MetaPathsInput':
        converted_meta_paths = []
        node_types = set()
        for meta_paths in json.keys():
            node_ids = [int(id) for id in meta_paths.split(seperator)]
            converted_meta_paths.append(node_ids)
            node_types |= set(node_ids)

        return cls(converted_meta_paths, node_types)

    def _apply_transformation(self, meta_paths):
        features = {'node': [], 'context': []}
        for paths in meta_paths:
            for node_key in range(len(paths)):
                node = paths[node_key]
                left_keys = self._left_context(node_key, self.window_size)
                right_keys = self._right_context(node_key + 1, len(paths), self.window_size)
                context_keys = np.array(left_keys + right_keys, dtype=np.int32) + 1
                context = np.array([self.padding_value - 1] + paths, dtype=np.int32) + 1
                context = context[context_keys]

                features['node'].append(node)
                features['context'].append(context)
        # Finally convert to array
        features['context'] = np.array(features['context'], np.int32)
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
        return tf.data.Dataset().from_tensor_slices(({'features': self.nodes}, self.contexts))

    def bag_of_words_input(self) -> tf.data.Dataset:
        """
        Get the dataset to train on in continuous bag of words format.
        :return: the dataset with context as features and nodes as labels.
        """
        self._update()
        return tf.data.Dataset().from_tensor_slices(({'features': self.contexts}, self.nodes))

    def get_vocab_size(self) -> Number:
        return len(self.nodes)

    def get_vocab(self) -> List[Number]:
        return self.nodes


class NodeInput(Input):
    pass


def model_word2vec(features, labels, mode, params):
    """
    Word2vec model from "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al.)
    TODO: Add documentation
    :return:
    """
    input = tf.feature_column.input_layer(features, params['feature_columns'])

    size_of_vocabulary = input.shape[1].value

    word_embeddings = tf.Variable(
        initial_value=tf.random_uniform(shape=[size_of_vocabulary, params['embedding_size']], minval=-1, maxval=1),
        name='word_embeddings')

    # Look up embedding for all words
    embedded_words = tf.nn.embedding_lookup(word_embeddings, tf.argmax(input, axis=1))

    return _model_word2vec(mode, size_of_vocabulary, params['loss'], labels, embedded_words)


def _model_word2vec(mode, size_of_vocabulary, loss: str, labels, embedded_words):
    # Concatenate vectors
    concatenated_embeddings = tf.concat(tf.unstack(embedded_words, axis=0), axis=0)

    # Transform embeddings linearly
    hidden_layer = tf.layers.dense(inputs=concatenated_embeddings, units=size_of_vocabulary, activation=None,
                                   use_bias=True,
                                   name="linear_transformation",
                                   kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

    # Apply softmax and calulate loss
    if loss == 'cross_entropy':
        labels = tf.one_hot(indices=labels, depth=size_of_vocabulary)
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


def create_estimator(model_dir, model_fn, input: Input, embedding_size: int, loss: str):
    features = tf.feature_column.categorical_column_with_identity('features',
                                                                  num_buckets=input.get_vocab_size())
    indicator_column = tf.feature_column.indicator_column(features)
    run_config = tf.estimator.RunConfig(tf_random_seed=42,
                                        save_summary_steps=500,
                                        save_checkpoints_steps=1000,
                                        keep_checkpoint_max=5,
                                        keep_checkpoint_every_n_hours=0.25,
                                        log_step_count_steps=50)
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params={'feature_columns': [indicator_column],
                'embedding_size': embedding_size,
                'loss': loss},
        config=run_config)
    return classifier


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--mode",
                        choices=["train", "predict", "eval"],
                        default="train",
                        help="Specify what you want to do: 'train', 'predict' or 'eval'.")
    parser.add_argument('--input_type',
                        choices=["meta-paths", "nodes"],
                        help='Choose input type of data in json',
                        type=str,
                        required=True)
    parser.add_argument('--json_path',
                        help='Specify path of json with input data',
                        type=str,
                        required=True)
    parser.add_argument('--model_dir',
                        help='Specify directory where checkpoints etc. should be saved',
                        type=str,
                        required=True)
    parser.add_argument('--model',
                        choices=["word2vec", "paragraph_vectors"],
                        help='Choose which model should be used',
                        type=str,
                        required=True)
    parser.add_argument('--model_type',
                        choices=["bag-of-words", "skip-gram"],
                        help='Choose which model type should be used',
                        type=str,
                        required=True)
    parser.add_argument('--embedding_size',
                        help='Specify the size of the embedding',
                        type=int,
                        required=True)
    parser.add_argument('--loss',
                        choices=["cross_entropy"],
                        help='Choose which loss should be used',
                        type=str,
                        required=True)
    return parser.parse_args()


def choose_function(model: str, model_type: str, input_type: str, json_path: str):
    json_file = open(json_path, mode='r')
    json_data = json.load(json_file)
    if input_type == "meta-paths":
        input = MetaPathsInput.from_json(json_data)
        if model_type == "bag-of-words":
            input.bag_of_words_input()
            input_fn = input.bag_of_words_input
        elif model_type == "skip-gram":
            input.skip_gram_input()
            input_fn = input.skip_gram_input
    elif input_type == "meta-nodes":
        input = NodeInput.from_json(json_data)
        if model_type == "bag-of-words":
            input.bag_of_words_input()
            input_fn = input.bag_of_words_input
        elif model_type == "skip-gram":
            input.skip_gram_input()
            input_fn = input.skip_gram_input
    if model == "word2vec":
        model_fn = model_word2vec
    elif model == "paragraph_vectors":
        if model_type == "bag-of-words":
            model_fn = model_paragraph_vectors_dbow
        elif model_type == "skip-gram":
            model_fn = model_paragraph_vectors_skipgram
    return input, model_fn, input_fn


if __name__ == "__main__":
    args = parse_arguments()
    input, model_fn, input_fn = choose_function(model=args.model, model_type=args.model_type,
                                                input_type=args.input_type,
                                                json_path=args.json_path)

    classifier = create_estimator(model_dir=args.model_dir, model_fn=model_fn, input=input,
                                  embedding_size=args.embedding_size, loss=args.loss)

    if args.mode == 'train':
        classifier.train(input_fn=input_fn)
    elif args.mode == 'predict':
        raise NotImplementedError()
    elif args.mode == 'eval':
        raise NotImplementedError()
