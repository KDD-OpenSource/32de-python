import random


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

    def paragraph_postprocess(self, paragraph_id, node_predict, context):
        raise NotImplementedError()

    def iterator(self, path_length, samples):
        raise NotImplementedError()

    def index(self, path, index):
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
        return None