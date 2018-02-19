from typing import List
from util.datastructures import MetaPathRating
from util.config import BASELINE_MODE

import numpy as np


class Explanation:
    def __init__(self):
        raise NotImplementedError()


class SimilarityScore:
    meta_paths = None
    similarity_score = None
    algorithm_type = None

    def __init__(self, algorithm_type=BASELINE_MODE):
        self.algorithm_type = algorithm_type

    '''
    Later, we will fetch meta-paths dynamically. At the moment, we pass meta paths directly
    from the tests.
    '''
    @staticmethod
    def fetch_meta_paths(self) -> List[MetaPathRating]:
        return self.meta_paths

    '''
    Computes a sum of a linear combination of structural and domain value
    over all meta-paths, normalized by each meta-path length. First simplified,
    not experimentally tested baseline.
    '''
    @staticmethod
    def calculate_similarity(meta_path_ratings: List[MetaPathRating]) -> float:
        structural_values = np.array([])
        domain_values = np.array([])

        for meta_path_rating in meta_path_ratings:
            structural_values = np.append(structural_values, [meta_path_rating.structural_value])
            domain_values = np.append(domain_values, [meta_path_rating.domain_value])

        return np.sum(structural_values * domain_values) / len(meta_path_ratings)
