import unittest
from util.datastructures import MetaPath
from explanation.explanation import SimilarityScore


class SimilarityTestCase(unittest.TestCase):
    sim_score = None

    def setUp(self):
        self.sim_score = SimilarityScore()

    def test_similarity_calculation(self):
        test_paths = [MetaPath([1, 2], [3], 3, 1), MetaPath([1, 2, 1], [3, 4], 1, 3), MetaPath([1, 3, 2], [1, 1], 5, 2),
                      MetaPath([1, 3, 2], [1, 1], 4, 7), MetaPath([1, 3, 2], [1, 1], 6, 3)]

        similarity = SimilarityScore.calculate_similarity(test_paths)

        self.assertEqual(similarity, 12.4)


if __name__ == '__main__':
    unittest.main()