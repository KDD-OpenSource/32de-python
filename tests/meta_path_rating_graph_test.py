# import unittest
# from typing import List
# from util.datastructures import MetaPathRatingGraph, MetaPath
# import numpy
#
# class MetaPathRatingGraphTest(unittest.TestCase):
#
#     def setUp(self):
#         self.meta_paths = self.create_meta_path_list()
#
#     def test_creation(self):
#         rating_graph = MetaPathRatingGraph()
#
#         self.assertIsNotNone(rating_graph)
#
#     def test_filling(self):
#         rating_graph = MetaPathRatingGraph()
#         pairs = [(1, 2)]
#         distances = [10]
#         self.fill_rating_graph(graph=rating_graph, pairs=pairs, distances=distances)
#
#         self.assertEqual(rating_graph.number_of_nodes(), 2)
#         self.assertEqual(rating_graph.number_of_edges(), 1)
#
#     def test_distance_extraction(self):
#         rating_graph = MetaPathRatingGraph()
#         pairs = [(1, 2), (2, 3), (3, 5), (1, 4)]
#         distances = [4.2, 6, 1, 3]
#         self.fill_rating_graph(graph=rating_graph, pairs=pairs, distances=distances)
#
#         expected_distances = {
#             self.meta_paths[1]: [0.0, 4.2, 10.2, 11.2, 3.0],
#             self.meta_paths[2]: [0.0, 6.0, 7.0],
#             self.meta_paths[3]: [0.0, 1],
#             self.meta_paths[5]: [0.0],
#             self.meta_paths[4]: [0.0]
#         }
#
#         rating_graph_result = rating_graph.all_pair_distances()
#
#         for node in rating_graph.all_nodes():
#             for distance in expected_distances[node]:
#                 self.assertIn(distance, list(rating_graph_result[rating_graph.meta_path_to_vertex[node]]))
#
#     def test_distance_streaming(self):
#         rating_graph = MetaPathRatingGraph()
#         pairs = [(1, 2), (2, 3), (3, 5), (1, 4)]
#         distances = [4.2, 6, 1, 3]
#         self.fill_rating_graph(graph=rating_graph, pairs=pairs, distances=distances)
#
#         expected_distances_triples = [
#             (self.meta_paths[1], self.meta_paths[2], 4.2),
#             (self.meta_paths[1], self.meta_paths[3], 10.2),
#             (self.meta_paths[1], self.meta_paths[5], 11.2),
#             (self.meta_paths[1], self.meta_paths[4], 3.0),
#             (self.meta_paths[2], self.meta_paths[3], 6.0),
#             (self.meta_paths[2], self.meta_paths[5], 7.0),
#             (self.meta_paths[3], self.meta_paths[5], 1.0)
#         ]
#
#         for i, triple in enumerate(rating_graph.stream_meta_path_distances()):
#             self.assertEqual(triple, expected_distances_triples[i])
#
#     def test_distance_streaming_zero_filter(self):
#         rating_graph = MetaPathRatingGraph()
#         pairs = [(1, 2), (2, 3), (3, 5), (1, 4)]
#         distances = [4.2, 6, 1, 3]
#         self.fill_rating_graph(graph=rating_graph, pairs=pairs, distances=distances)
#
#         for triple in rating_graph.stream_meta_path_distances():
#             self.assertNotEqual(0, triple[2])
#
#     def test_distance_streaming_infinity_filter(self):
#         rating_graph = MetaPathRatingGraph()
#         pairs = [(1, 2), (2, 3), (3, 5), (1, 4)]
#         distances = [4.2, 6, 1, 3]
#         self.fill_rating_graph(graph=rating_graph, pairs=pairs, distances=distances)
#
#         for triple in rating_graph.stream_meta_path_distances():
#             self.assertNotEqual(numpy.inf, triple[2])
#
#     def test_get_metapaths(self):
#         rating_graph = MetaPathRatingGraph()
#         pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
#         distances = [1, 1, 1, 1, 1]
#         self.fill_rating_graph(graph=rating_graph, pairs=pairs, distances=distances)
#
#         self.assertEqual(set(rating_graph.all_nodes()), set(self.meta_paths))
#
#     def create_meta_path_list(self) -> List[MetaPath]:
#         return [
#             MetaPath(nodes=['SNP', 'PHN', 'SNP'], edges=['has', 'married_to']),
#             MetaPath(nodes=['PHN', 'City'], edges=['lives_in']),
#             MetaPath(nodes=['City', 'SNP'], edges=['has']),
#             MetaPath(nodes=['GEN', 'PHN', 'GEN'], edges=['married_to', 'lives_in']),
#             MetaPath(nodes=['PHN', 'SNP', 'SNP'], edges=['married_to', 'married_to']),
#             MetaPath(nodes=['Person', 'SNP'], edges=['likes'])
#         ]
#
#     def fill_rating_graph(self, graph, pairs, distances):
#         for pair, distance in zip(pairs, distances):
#             graph.add_user_rating(self.meta_paths[pair[0]], self.meta_paths[pair[1]], distance)
#
#
# if __name__ == '__main__':
#     unittest.main()
