import random


class SamplingStrategy():
    """
    Strategy for sampling word and paragraph2vec training data depending on
    training type (bag of words or skip gram).
    """

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
        """
        Used for node or equivalent embeddings. Sample the context for the node that is focused.
        In bag of words this is the node that should be predicted, in skip gram the one to be
        embedded.
        :param node_key: The index of the focused node.
        :param path_length: The total length of the body (for example the path)
                            the word is sampled from.
        :param window_size: The number of context nodes to be samples from each side.
        :return: The indices of the nodes that should be used as the context.
        """
        raise NotImplementedError()

    def sample_paragraph(self, node_key, path_length, window_size):
        """
        Used for path or equivalent embeddings. Sample the context for the path that should be embedded.
        :param node_key: This is the index of the node that should be predicted (not necessary in all methods).
        :param path_length: The total length of the path.
        :param window_size: The number of context nodes to be samples from each side. If the sampling is done
                            from the whole path this is **half** the number of context nodes sampled.
        :return: The indices of the nodes that should be used as the context.
        """
        raise NotImplementedError()

    def paragraph_postprocess(self, paragraph_ids, node_predictions, contexts):
        """
        Used for path or equivalent embeddings. After having sampled the training data,
        paragraph specific post-processing needs to be done.
        :param paragraph_ids: The ids of the paragraphs the sample belongs to.
        :param node_predictions: The focused nodes of each training sample.
        :param contexts: The context nodes of each training sample.
        :return: The preprocessed lists of the input data.
        """
        raise NotImplementedError()

    def iterator(self, path_length, samples):
        """
        Used for path or equivalent embeddings. Calculates the number of samples that should be created for each path.
        :param path_length: The total length of the path being processed.
        :param samples: The number of samples the user has specified.
        :return: The number of samples that should be created for each path.
        """
        raise NotImplementedError()

    def index(self, path, index):
        """
        Used for path or equivalent embeddings. Finds the focus node if there is one.
        :param path: The actual path to find the focus node in.
        :param index: The current sampling iteration.
        :return: If there is a focus node, it is returned, None otherwise.
        """
        raise NotImplementedError()


class CBOWSampling(SamplingStrategy):

    def sample_word(self, node_key, _, window_size):
        window_size = 2 * window_size
        return self._primitive_context(list(range(max(1, node_key - window_size), node_key)), window_size)

    def sample_paragraph(self, node_key, path_length, window_size):
        return self.sample_word(node_key, path_length, window_size)

    def paragraph_postprocess(self, paragraph_id, node_predict, context):
        return paragraph_id, context, node_predict

    def iterator(self, path_length, samples):
        return range(1, path_length)

    def index(self, path, index):
        return path[index]


class SkipGramSampling(SamplingStrategy):

    def sample_word(self, node_key, path_length, window_size):
        left_keys = self._left_context(node_key, window_size)
        right_keys = self._right_context(node_key, path_length, window_size)
        return left_keys + right_keys

    def sample_paragraph(self, node_key, path_length, window_size):
        return self._primitive_context(list(range(1, path_length)), 2 * window_size)

    def paragraph_postprocess(self, paragraph_id, node_predict, context):
        return paragraph_id, context, []

    def iterator(self, path_length, samples):
        return range(samples)

    def index(self, path, index):
        return 0