import unittest
from embeddings.sampling_strategy import CBOWSampling, SkipGramSampling


class MetaPathsInputTest(unittest.TestCase):

    def test_cbow_strategy(self):
        sampler = CBOWSampling(0, 42)
        self.assertEqual([0, 0, 1, 2], sampler.sample_word(3, 6, 2))

    def test_skip_gram_strategy(self):
        sampler = SkipGramSampling(0, 42)
        self.assertEqual([1, 2, 4, 5], sampler.sample_word(3, 6, 2))

    def test_cbow_strategy_upper(self):
        sampler = CBOWSampling(0, 42)
        self.assertEqual([3, 6, 4, 5], sampler.sample_word(7, 8, 2))

if __name__ == '__main__':
    unittest.main()
