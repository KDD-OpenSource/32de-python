from numbers import Number
from typing import List

import numpy as np
import random
import tensorflow as tf

from embeddings.sampling_strategy import CBOWSampling, SkipGramSampling, SamplingStrategy


class Input:

    def __init__(self,
                 paths: List[List[Number]],
                 vocabulary: List[Number],
                 windows_size: Number = 2,
                 padding_value: Number = -1,
                 random_seed: Number = 42,
                 normalize_node_ids=True):
        self.mapping = {}
        self.inverse_mapping = {}
        self.padding_index = 0
        self.padding_mapping = 0
        self.padding_value = padding_value
        self.random_seed = random_seed
        self.window_size = windows_size
        self.samplingStrategies = {
            'cbow': CBOWSampling(self.padding_index, self.random_seed),
            'skip-gram': SkipGramSampling(self.padding_index, self.random_seed)
        }

        if (normalize_node_ids):
            self.paths = self._normalize_node_ids(paths, list(vocabulary))
        else:
            self.paths = paths

    def _normalize_node_ids(self, meta_paths, vocabulary):
        normalized_paths = []
        mapped_values = range(1, len(vocabulary) + 1)
        self.mapping = dict(zip(vocabulary, mapped_values))
        self.inverse_mapping = dict(zip(mapped_values, vocabulary))

        for mp in meta_paths:
            path = []
            path.append(self.padding_mapping)
            for n in mp:
                path.append(self.mapping[n])
                normalized_paths.append(path)

        return normalized_paths

    @classmethod
    def extract_paths_vocab_json(cls, json, seperator):
        converted_paths = []
        vocabulary = set()
        for paths in json.keys():
            node_ids = [int(id) for id in paths.split(seperator)]
            converted_paths.append(node_ids)
            vocabulary |= set(node_ids)
        return converted_paths, vocabulary

    @classmethod
    def from_json(cls, json, seperator=" | ") -> 'Input':
        converted_paths, vocabulary = cls.extract_paths_vocab_json(json, seperator)

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
        node, context = self._apply_transformation(self.paths, self.samplingStrategies['skip-gram'])
        return self._create_dataset(np.reshape(node, (-1, 1)), context)

    def bag_of_words_input(self) -> tf.data.Dataset:
        """
        Get the dataset to train on in continuous bag of words format.
        :return: the dataset with context as features and node types as labels.
        """
        node, context = self._apply_transformation(self.paths, self.samplingStrategies['cbow'])
        return self._create_dataset(context, node)

    def _create_dataset(self, features, labels):
        return tf.data.Dataset().from_tensor_slices(({'features': features}, labels))

    def get_vocab_size(self) -> int:
        return len(self.get_vocab())

    def get_vocab(self) -> List[Number]:
        return list(self.inverse_mapping.keys())

    def get_node_id(self, mapped_id):
        return self.inverse_mapping[mapped_id]

    def get_mapped_id(self, node_id):
        return self.mapping[node_id]

    def paths_count(self):
        return len(self.paths)

    def _apply_transformation(self, meta_paths: List[List[Number]], sampling_strategy: SamplingStrategy):
        raise NotImplementedError()


class NodeEdgeTypeInput(Input):

    def _apply_transformation(self, meta_paths: List[List[Number]], sampling_strategy: SamplingStrategy):
        nodes = []
        contexts = []

        for path in meta_paths:
            for node_key in range(1, len(path)):
                node = path[node_key]
                context = np.array(path, dtype=np.int32)[sampling_strategy.sample_word(node_key,
                                                                                       len(path),
                                                                                       self.window_size)]

                nodes.append(node)
                contexts.append(context)
        # Finally convert to array
        contexts = np.array(contexts, np.int32)
        nodes = np.array(nodes, np.int32)
        return nodes, contexts


class MetaPathsInput(Input):

    def __init__(self,
                 paths: List[List[Number]],
                 vocabulary: List[Number] = [],
                 windows_size: Number = 2,
                 padding_value: Number = 0,
                 samples: Number = 5,
                 random_seed: Number = 42,
                 normalize_node_ids=False):
        """
        Wraps in the input function to embed meta-paths. It is a convention that the id of the meta-paths is given
        by it's location in the paths list. Therefore the order of the paths is important.
        """
        super().__init__(paths, vocabulary, windows_size, padding_value, random_seed, normalize_node_ids)
        self.samples = samples

    @classmethod
    def from_json(cls, json, seperator="|"):
        paths = [path.split(seperator) for path in json]
        return cls.from_paths_list(paths)

    @classmethod
    def from_paths_list(cls, paths: List[List[int]]):
        # The ordering of the paths and converted_paths has to be identical to allow remapping of the embedding to the
        # original path.
        converted_paths = []

        mapping = {}
        inverse_mapping = {}

        padding_value = 0
        highest_key = 1

        for path in paths:
            assert len(path) % 2 == 1, "Invalid path shape."
            new_path = []
            if len(path) == 1:
                new_path.append(path[0])
                converted_paths.append(new_path)
                continue
            for i in range(0, len(path) - 2, 2):
                node_edge_node = (path[i], path[i + 1], path[i + 2])
                key = highest_key
                if node_edge_node in mapping.keys():
                    key = mapping[node_edge_node]
                else:
                    mapping[node_edge_node] = key
                    inverse_mapping[key] = node_edge_node
                    highest_key += 1
                new_path.append(key)
            converted_paths.append(new_path)

        input_cls = cls(converted_paths, padding_value=padding_value, normalize_node_ids=False)

        input_cls.mapping = mapping
        input_cls.inverse_mapping = inverse_mapping

        return input_cls

    def set_samples(self, samples: Number) -> 'MetaPathsInput':
        # TODO: Max this value for each mp to (len(mp) over window_size)
        """
        Set the number samples. For each meta-paths random node samples are taken, this is repeated *samples* times.
        :param samples: the number of training samples to create for each meta-paths.
        :return: Input object itself.
        """
        self.samples = samples
        return self

    def _apply_transformation(self, meta_paths: List[List[Number]], sampling_strategy: SamplingStrategy):
        paths = []
        indices = []
        contexts = []

        path_id = 0
        for path in meta_paths:
            for iteration in sampling_strategy.iterator(len(path), self.samples):
                context = np.array(path, dtype=np.int32)[sampling_strategy.sample_paragraph(iteration,
                                                                                            len(path),
                                                                                            self.window_size)]

                paths.append(path_id)
                indices.append(sampling_strategy.index(path, iteration))
                contexts.append(context)
            path_id += 1

        paths = np.array(paths, np.int32)
        indices = np.array(indices, np.int32)
        contexts = np.array(contexts, np.int32)
        return sampling_strategy.paragraph_postprocess(paths, indices, contexts)

    def skip_gram_input(self) -> tf.data.Dataset:
        """
        Get the dataset to train on in skip-gram format.
        :return: the dataset with node types as features and context as labels.
        """
        paragraphs, context, _ = self._apply_transformation(self.paths, self.samplingStrategies['skip-gram'])
        return self._create_dataset(paragraphs, context)

    def bag_of_words_input(self) -> tf.data.Dataset:
        """
        Get the dataset to train on in continuous bag of words format.
        :return: the dataset with context as features and node types as labels.
        """
        paragraphs, context, node = self._apply_transformation(self.paths, self.samplingStrategies['cbow'])
        return self._create_dataset(paragraphs, context, node)

    def _create_dataset(self, paragraphs, features, labels):
        print(paragraphs)
        return tf.data.Dataset().from_tensor_slices(({
                                                         'features': features,
                                                         'paragraphs': np.reshape(paragraphs, (-1, 1))
                                                     },
                                                     labels))


UNDEFINED_SYMBOL = -1

class NodeInput(Input):

    def __init__(self,
                 dataset: tf.data.Dataset,
                 num_skips: Number,
                 skip_window: Number,
                 vocabulary: List[Number] = [],
                 normalize_node_ids=False):
        self.dataset = dataset
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.vocabulary = vocabulary

    @classmethod
    def from_raw_file(cls, file_names: List[str], num_skips: Number, skip_window: Number):
        dataset = tf.data.TextLineDataset(file_names)

        dataset = dataset.map(lambda line: tf.string_split([line]).values)

        # default_value = str(UNDEFINED_SYMBOL)
        # dataset = dataset.map(lambda sparse_tensor:
        #                       (tf.sparse_tensor_to_dense(sparse_tensor, default_value=default_value)))

        # dataset = dataset.map(lambda tensor: (tensor.set_shape([tf.size(tensor).value])))

        # iterator = dataset.make_one_shot_iterator()
        # print(tf.get_shape(iterator.get_next()))

        dataset = dataset.map(lambda string_tensor: (tf.string_to_number(string_tensor, out_type=tf.int32)))

        return cls(dataset, num_skips, skip_window)

    @staticmethod
    def convert_line(line):
        string_list = tf.string_split([line])
        print(string_list.eval())
        return string_list

    def skip_gram_input(self) -> tf.data.Dataset:
        """
        Get the dataset to train on in skip-gram format.
        :return: the dataset with node types as features and context as labels.
        """
        # output_labels = tf.reshape(self.dataset, [-1])
        output_labels = self.dataset.map(lambda labels:tf.reshape(labels, [-1, 1]))

        output_context = self.dataset.map(lambda walk: (self.create_context(walk, self.skip_window, self.num_skips)))
        output_together = tf.data.Dataset.zip((output_labels, output_context))
        return output_together

    def create_context(self, tensor, skip_window, num_skips):
        with_padding = self.add_padding_for_tensor(tensor, skip_window)
        context_shape = [1, tensor.shape[1].value, num_skips]
        context_tensor = tf.zeros(context_shape, dtype=tf.int32)
        for i in range(0, tensor.shape[1].value):
            with_padding_index = i + skip_window
            context = self.create_context_for_index(with_padding_index, skip_window, num_skips, with_padding)
            context_tensor = tf.scatter_update(context_tensor, [i], context)
        return context_tensor

    @staticmethod
    def add_padding_for_tensor(tensor, padding_size):
        paddings = tf.constant([[1, padding_size]])
        constant_values = tf.constant([UNDEFINED_SYMBOL])
        # print(tensor.eval())
        return tf.pad(tensor, paddings, "CONSTANT", constant_values=constant_values)
        # padding = tf.tile(PADDING_SYMBOL, padding_size)
        # return tf.concat([padding, tensor, padding], 0)

    def create_context_for_index(self, index, skip_window, num_skips, tensor):
        front_context = tensor[index - skip_window:index]
        back_context = tensor[index:index + skip_window]
        full_context = tf.concat([front_context, back_context], 0)

        window_size = full_context.shape[1].value
        # TODO only use unique indices, e.g. instead of [1, 1, 6, 1] use [2, 5, 1, 6]
        selected_indices = tf.random_uniform([1, num_skips], maxval=window_size, dtype=tf.int32)
        selected_context = tf.gather(tensor, selected_indices)
        return selected_context

    def bag_of_words_input(self) -> tf.data.Dataset:
        raise NotImplementedError()

    def get_vocab_size(self) -> int:
        return len(self.get_vocab())

    pass

print("hello")
sess = tf.InteractiveSession()
node_input = NodeInput.from_raw_file(["mock_input.txt"], 2, 2)
skip_gram_dataset = node_input.skip_gram_input()
sess.close()