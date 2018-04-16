import unittest
from embeddings.input import NodeEdgeTypeInput


class MetaPathsInputTest(unittest.TestCase):

    def setUp(self):
        self.mps = [[1, 2, 1], [1, 10, 2, 1], [1, 11, 10, 11, 1]]
        self.vocab = [1, 10, 2, 11]
        self.input = NodeEdgeTypeInput(paths=self.mps,
                                       vocabulary=self.vocab)
    def test_correct_vocabulary_size(self):
        self.assertEqual(len(self.vocab), self.input.get_vocab_size())

    def test_correct_vocabulary(self):
        self.assertEqual([1, 2, 3, 4], self.input.get_vocab())

    def test_correct_vocabulary_mapping(self):
        self.assertEqual(1, self.input.get_node_id(1))
        self.assertEqual(10, self.input.get_node_id(2))
        self.assertEqual(2, self.input.get_node_id(3))
        self.assertEqual(11, self.input.get_node_id(4))

    def test_correct_normalization(self):
        self.assertEqual([0, 1, 3, 1], self.input.paths[0])

if __name__ == '__main__':
    unittest.main()
