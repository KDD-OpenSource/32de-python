import unittest
from domain_scoring.domain_scoring import DomainScoring
from util.ranking_graph import RankingGraph
from util.datastructures import MetaPath

class DomainScoringTest(unittest.TestCase):

    def setUp(self):
        self.ds = DomainScoring()

        self.ranking_graph = RankingGraph()
        self.ranking_graph.transitive_closures = lambda: [[1, "B", "C"], ["B", "C"]]
        self.ranking_graph.all_nodes = lambda: [1, "B", "C"]

    def test_extract_features_labels(self):
        self.assertEqual(
            ([(1, "B"), ("B", 1), (1, "C"), ("C", 1), ("B", "C"), ("C", "B")], [0, 1, 0, 1, 0, 1]),
            self.ds._extract_training_data_labels(self.ranking_graph))

    def test_fit_vectorizer(self):
        self.assertIsNone(self.ds._fit_vectorizer(self.ranking_graph))


    def test_preprocess(self):
        expected = [[0, 0, 1, 1, 0, 0]]

        metapaths = [
            MetaPath(["C"]),
            MetaPath([1])
        ]

        metapaths_tuples = [(metapaths[0], metapaths[1])]

        self.ds._fit_vectorizer(self.ranking_graph)
        self.assertEqual(expected, self.ds._preprocess(metapaths_tuples))

    def test_fit(self):
        self.assertIsNone(self.ds.fit(self.ranking_graph))



if __name__ == '__main__':
    unittest.main()