import unittest
from util.datastructures import MetaPath, MetaPathRating
from explanation.explanation import SimilarityScore


class SimilarityTestCase(unittest.TestCase):
    sim_score = None

    def setUp(self):
        self.sim_score = SimilarityScore()

    def test_similarity_calculation(self):
        test_paths = \
            [MetaPathRating(MetaPath([1, 2], [3]), 3, 1), MetaPathRating(MetaPath([1, 2, 1], [3, 4]), 1, 3),
             MetaPathRating(MetaPath([1, 3, 2], [1, 1]), 5, 2), MetaPathRating(MetaPath([1, 3, 2], [1, 1]), 4, 7),
             MetaPathRating(MetaPath([1, 3, 2], [1, 1]), 6, 3)]

        similarity = SimilarityScore.calculate_similarity(test_paths)

        self.assertEqual(similarity, 12.4)


if __name__ == '__main__':
    unittest.main()
