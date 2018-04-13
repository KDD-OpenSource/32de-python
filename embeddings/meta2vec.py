from numbers import Number
from typing import Tuple, List
import tensorflow as tf
import numpy as np
import random
import argparse
import json


class SamplingStrategy():

    def __init__(self, padding_index, random_seed):
        self.random_seed = random_seed
        self.padding_index = padding_index

    def _left_context(self, max_key, win_size):
        return self._primitive_context(list(range(1, max_key)), win_size)

    def _right_context(self, min_key, max_key, win_size):
        return self._primitive_context(list(range(min_key + 1, max_key)), win_size)

    def _primitive_context(self, key_range, windows_size):
        if len(key_range) < windows_size:
            return [self.padding_index] * (windows_size - len(key_range)) + key_range
        random.seed(self.random_seed)
        return random.sample(key_range, windows_size)

    def sample_word(self, node_key, path_length, window_size):
        raise NotImplementedError()

    def sample_paragraph(self, node_key, path_length, window_size):
        raise NotImplementedError()

    def iterator(self, path_length, samples):
        raise NotImplementedError()


class SkipGramSampling(SamplingStrategy):

    def sample_word(self, node_key, _, window_size):
        return self._primitive_context(list(range(max(1, node_key - window_size), node_key)), 2 * window_size)

    def sample_paragraph(self, node_key, path_length, window_size):
        return self.sample_word(node_key, path_length, window_size)

    def iterator(self, path_length, samples):
        return range(1, len(path_length))


class CBOWSampling(SamplingStrategy):

    def sample_word(self, node_key, path_length, window_size):
        left_keys = self._left_context(node_key, window_size)
        right_keys = self._right_context(node_key, path_length, window_size)
        return left_keys + right_keys

    def sample_paragraph(self, node_key, path_length, window_size):
        return self._primitive_context(list(range(1, path_length)), 2 * window_size)

    def iterator(self, path_length, samples):
        return range(samples)


class Input:

    def __init__(self,
                 paths: List[List[Number]],
                 vocabulary: List[Number],
                 windows_size: Number = 2,
                 padding_value: Number = -1,
                 random_seed: Number = 42):
        self.vocabulary = [padding_value] + list(vocabulary)
        self.padding_index = 0
        self.padding_value = padding_value
        self.random_seed = random_seed
        self.window_size = windows_size
        self.samplingStrategies = {
            'cbow': CBOWSampling(self.padding_index, self.random_seed),
            'skip-gram': SkipGramSampling(self.padding_index, self.random_seed)
        }

        self.paths = self._normalize_node_ids(paths)

    def _normalize_node_ids(self, meta_paths):
        normalized_paths = []
        id_mapping = dict(zip(self.vocabulary, self.get_vocab()))

        for mp in meta_paths:
            path = []
            path.append(id_mapping[self.padding_value])
            for n in mp:
                path.append(id_mapping[n])
                normalized_paths.append(path)

        return normalized_paths

    @classmethod
    def from_json(cls, json, seperator=" | ") -> 'Input':
        converted_paths = []
        vocabulary = set()
        for paths in json.keys():
            node_ids = [int(id) for id in paths.split(seperator)]
            converted_paths.append(node_ids)
            vocabulary |= set(node_ids)

        return cls(converted_paths, vocabulary)


    def set_window_size(self, window_size: Number) -> 'NodeEdgeTypeInput':
        """
        Set the window size. This number of nodes are each taken from the left and right of the feature node
        for the context.
        :param window_size: the value it should be set to.
        :return: the object itself.
        """
        self.window_size = window_size
        return self

    def set_padding_value(self, padding_value: Number) -> 'NodeEdgeTypeInput':
        """
        Set the padding value. With this value meta-paths will be padded at the beginning and end. This value will,
        for example, occur when the first or last node context is extracted.
        :param padding_value: the value it should be set to.
        :return: the object itself.
        """
        self.padding_value = padding_value
        return self

    def skip_gram_input(self) -> tf.data.Dataset:
        """
        Get the dataset to train on in skip-gram format.
        :return: the dataset with node types as features and context as labels.
        """
        embedding_value, context = self._apply_transformation(self.paths, self.samplingStrategies['skip-gram'])
        return self._create_dataset(np.reshape(embedding_value, (-1, 1)), context)

    def bag_of_words_input(self) -> tf.data.Dataset:
        """
        Get the dataset to train on in continuous bag of words format.
        :return: the dataset with context as features and node types as labels.
        """
        embedding_value, context = self._apply_transformation(self.paths, self.samplingStrategies['cbow'])
        return self._create_dataset(context, embedding_value)

    def _create_dataset(self, features, labels):
        return tf.data.Dataset().from_tensor_slices(({'features': features}, labels))

    def get_vocab_size(self) -> Number:
        return len(self.vocabulary)

    def get_vocab(self) -> List[Number]:
        return list(range(self.get_vocab_size()))

    def get_node_id(self, mapped_id):
        return self.vocabulary[mapped_id]

    def _apply_transformation(self, meta_paths: List[List[Number]], key_strategy: SamplingStrategy):
        raise NotImplementedError()


class NodeEdgeTypeInput(Input):

    def _apply_transformation(self, meta_paths: List[List[Number]], key_strategy: SamplingStrategy):
        features = {'node': [], 'context': []}
        for path in meta_paths:
            for node_key in range(1, len(path)):
                node = path[node_key]
                context = np.array(path, dtype=np.int32)[key_strategy.sample_word(node_key,
                                                                                  len(path),
                                                                                  self.window_size)]

                features['node'].append(node)
                features['context'].append(context)
        # Finally convert to array
        features['context'] = np.array(features['context'], np.int32)
        features['node'] = np.array(features['node'], np.int32)
        return features['node'], features['context']


class MetaPathsInput(Input):

    def __init__(self,
                 paths: List[List[Number]],
                 vocabulary: List[Number],
                 windows_size: Number = 2,
                 padding_value: Number = 0,
                 samples: Number = 5,
                 random_seed: Number = 42):
        super().__init__(paths, vocabulary, windows_size, padding_value, random_seed)
        self.samples = samples

    def set_samples(self, samples: Number) -> 'MetaPathsInput':
        # TODO: Max this value for each mp to (len(mp) over window_size)
        """
        Set the number samples. For each meta-paths random node samples are taken, this is repeated *samples* times.
        :param samples: the number of training samples to create for each meta-paths.
        :return: Input object itself.
        """
        self.samples = samples
        return self



    def _apply_transformation(self, meta_paths: List[List[Number]], key_strategy: SamplingStrategy):
        features = {'path': [], 'index': [], 'context': []}
        path_id = 0
        for path in meta_paths:
            for iteration in key_strategy.iterator(len(path), self.samples):
                context = np.array(path, dtype=np.int32)[key_strategy.sample_paragraph(iteration,
                                                                                       len(path),
                                                                                       self.window_size)]

                features['path'].append(path_id)
                features['index'].append(iteration)
                features['context'].append(context)
            path_id += 1
        # Finally convert to array
        features['context'] = np.array(features['context'], np.int32)
        features['path'] = np.array(features['path'], np.int32)
        return features['path'], features['context']

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
    assert embedded_words.shape == (
        input.shape[0].value, params['embedding_size']), 'Shape expected ({}, {}), but was {}'.format(
        input.shape[0].value, params['embedding_size'], embedded_words.shape)

    return _model_word2vec(mode, size_of_vocabulary, params['loss'], labels, embedded_words)


def _model_word2vec(mode, size_of_vocabulary, loss: str, labels, embedded_words):
    # Concatenate vectors
    concatenated_embeddings = tf.reshape(tf.concat(tf.unstack(embedded_words, axis=0), axis=0), shape=[1, -1])
    assert concatenated_embeddings.shape == (
        1, embedded_words.shape[0].value * embedded_words.shape[1].value), 'Shape expected ({}), but was {}'.format(
        1, embedded_words.shape[0].value * embedded_words.shape[1].value, concatenated_embeddings.shape)

    # Transform embeddings linearly
    hidden_layer = tf.layers.dense(inputs=concatenated_embeddings, units=size_of_vocabulary, activation=None,
                                   use_bias=True,
                                   name="linear_transformation",
                                   kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
    assert hidden_layer.shape == (
        1, size_of_vocabulary), 'Shape expected ({}, {}), but was {}'.format(
        1, size_of_vocabulary, hidden_layer.shape)

    # Apply softmax and calulate loss
    if loss == 'cross_entropy':
        print(labels)
        labels = tf.reshape(tf.one_hot(indices=labels, depth=size_of_vocabulary), shape=[1, -1])
        print(labels)
        assert labels.shape == (
            1, size_of_vocabulary), 'Shape expected ({}, {}), but was {}'.format(
            1, size_of_vocabulary, labels.shape)
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


def create_estimator(model_dir, model_fn, input: Input, embedding_size: int, loss: str, gpu_memory: float):
    features = tf.feature_column.categorical_column_with_hash_bucket('features',
                                                                     input.get_vocab_size(),
                                                                     dtype=tf.int32)
    indicator_column = tf.feature_column.indicator_column(features)
    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
    run_config = tf.estimator.RunConfig(tf_random_seed=42,
                                        save_summary_steps=500,
                                        save_checkpoints_steps=1000,
                                        keep_checkpoint_max=5,
                                        keep_checkpoint_every_n_hours=0.25,
                                        log_step_count_steps=50,
                                        session_config=session_config)
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
    parser.add_argument('--gpu_memory',
                        help='Specify amount of GPU memory this process is allowed to use',
                        type=float,
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
        input = NodeEdgeTypeInput.from_json(json_data)
        if model_type == "bag-of-words":
            input_fn = input.bag_of_words_input
        elif model_type == "skip-gram":
            input_fn = input.skip_gram_input
    elif input_type == "nodes":
        input = NodeInput.from_json(json_data)
        if model_type == "bag-of-words":
            input_fn = input.bag_of_words_input
        elif model_type == "skip-gram":
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
                                  embedding_size=args.embedding_size, loss=args.loss, gpu_memory=args.gpu_memory)

    if args.mode == 'train':
        classifier.train(input_fn=input_fn)
    elif args.mode == 'predict':
        raise NotImplementedError()
    elif args.mode == 'eval':
        raise NotImplementedError()
