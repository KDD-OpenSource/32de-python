from numbers import Number
from typing import List

import numpy as np
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

    def get_vocab_size(self) -> Number:
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


class NodeInput(Input):
    pass
