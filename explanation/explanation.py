from util.datastructures import MetaPath

import numpy as np

# mock data
mock_paths = [MetaPath([1, 2], [3], 3, 1), MetaPath([1, 2, 1], [3, 4], 1, 3), MetaPath([1, 3, 2], [1, 1], 5, 2),
			  MetaPath([1, 3, 2], [1, 1], 4, 7), MetaPath([1, 3, 2], [1, 1], 6, 3)]

# algorithm types
BASELINE = 'baseline'


class Explanation:
	def __init__(self):
		raise NotImplementedError()

class SimilarityScore():
	meta_paths = None
	similarity_score = None

	def __init__(self, algorithm_type=BASELINE):
		self.meta_paths = self.fetch_meta_paths()
		self.algorithm_type = algorithm_type

	def display_similarity_score(self):
		if self.similarity_score is None:
			raise ValueError("Compute similarity score first!")

		print("Similarity Score: {}".format(self.similarity_score))

	def compute_similarity_score(self):
		structural_scores = np.array([])
		domain_scores = np.array([])

		for meta_path in self.meta_paths:
			structural_scores = np.append(structural_scores, [meta_path.structural_score])
			domain_scores = np.append(domain_scores, [meta_path.domain_score])

		self.similarity_score = np.sum(structural_scores * domain_scores) / len(self.meta_paths)

	def fetch_meta_paths(self):
		return mock_paths
