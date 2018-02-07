from util.datastructures import MetaPath, MetaPathRating

import numpy as np

# mock data
mock_paths = [MetaPathRating(MetaPath([1, 2], [3]), 3, 1), MetaPathRating(MetaPath([1, 2, 1], [3, 4]), 1, 3),
              MetaPathRating(MetaPath([1, 3, 2], [1, 1]), 5, 2), MetaPathRating(MetaPath([1, 3, 2], [1, 1]), 4, 7),
              MetaPathRating(MetaPath([1, 3, 2], [1, 1]), 6, 3)]

# algorithm types
BASELINE = 'baseline'


class Explanation:
    def __init__(self):
        raise NotImplementedError()


class SimilarityScore:
    meta_paths = None
    similarity_score = None

    def __init__(self, algorithm_type=BASELINE):
        self.algorithm_type = algorithm_type

    @staticmethod
    def fetch_meta_paths(self):
        return mock_paths

    @staticmethod
    def calculate_similarity(meta_path_ratings):
        structural_values = np.array([])
        domain_values = np.array([])

        for meta_path_rating in meta_path_ratings:
            structural_values = np.append(structural_values, [meta_path_rating.structural_value])
            domain_values = np.append(domain_values, [meta_path_rating.domain_value])

        return np.sum(structural_values * domain_values) / len(meta_path_ratings)


