import unittest
from embeddings.meta2vec import *

class MetaPathsInputTest(unittest.TestCase):

    def test_cbow_key_strategy(self):
        sampler = CBOWSampling(0, 42)
        self.assertEqual([0, 0, 1, 2], sampler.sample_word(3, 6, 2))

    def test_skip_gram_key_strategy(self):
        sampler = SkipGramSampling(0, 42)
        self.assertEqual([1, 2, 4, 5], sampler.sample_word(3, 6, 2))

if __name__ == '__main__':
    unittest.main()
