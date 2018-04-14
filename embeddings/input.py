from numbers import Number
from typing import List

import numpy as np
#import tensorflow as tf

from embeddings.sampling_strategy import CBOWSampling, SkipGramSampling, SamplingStrategy


class Input:

    def __init__(self,
                 paths: List[List[Number]],
                 vocabulary: List[Number],
                 windows_size: Number = 2,
                 padding_value: Number = -1,
                 random_seed: Number = 42):
        self.vocabulary = list(vocabulary)
        self.padding_index = 0
        self.padding_mapping = 0
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
            path.append(self.padding_mapping)
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

    def skip_gram_input(self): #-> tf.data.Dataset:
        """
        Get the dataset to train on in skip-gram format.
        :return: the dataset with node types as features and context as labels.
        """
        node, context = self._apply_transformation(self.paths, self.samplingStrategies['skip-gram'])
        return self._create_dataset(np.reshape(node, (-1, 1)), context)

    def bag_of_words_input(self): #-> tf.data.Dataset:
        """
        Get the dataset to train on in continuous bag of words format.
        :return: the dataset with context as features and node types as labels.
        """
        node, context = self._apply_transformation(self.paths, self.samplingStrategies['cbow'])
        return self._create_dataset(context, node)

    def _create_dataset(self, features, labels):
        return None #tf.data.Dataset().from_tensor_slices(({'features': features}, labels))

    def get_vocab_size(self) -> Number:
        return len(self.vocabulary)

    def get_vocab(self) -> List[Number]:
        return list(range(1, self.get_vocab_size() + 1))

    def get_node_id(self, mapped_id):
        return self.vocabulary[mapped_id - 1]

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

    def skip_gram_input(self): #-> tf.data.Dataset:
        """
        Get the dataset to train on in skip-gram format.
        :return: the dataset with node types as features and context as labels.
        """
        paragraphs, context, _ = self._apply_transformation(self.paths, self.samplingStrategies['skip-gram'])
        return self._create_dataset(np.reshape(paragraphs, (-1, 1)), context)

    def bag_of_words_input(self): #-> tf.data.Dataset:
        """
        Get the dataset to train on in continuous bag of words format.
        :return: the dataset with context as features and node types as labels.
        """
        paragraphs, context, node = self._apply_transformation(self.paths, self.samplingStrategies['cbow'])
        return self._create_dataset(paragraphs, context, node)

    def _create_dataset(self, paragraphs, features, labels):
        return None #tf.data.Dataset().from_tensor_slices(({
                    #                                     'features': features,
                    #                                     'paragraphs': paragraphs
                    #                                 },
                    #                                 labels))


class NodeInput(Input):
    pass
