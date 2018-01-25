import unittest
from typing import List, Iterable
from util.datastructures import MetaPathRatingGraph, MetaPath
import numpy

class MetaPathRatingTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MetaPathRatingTest, self).__init__(*args, **kwargs)
        self.meta_paths = self.create_meta_path_list()

    def test_creation(self):
        rating_graph = MetaPathRatingGraph()

        self.assertTrue(rating_graph is not None)

    def test_filling(self):
        rating_graph = MetaPathRatingGraph()
        pairs = [(1,2)]
        distances = [10]
        self.fill_rating_graph(graph=rating_graph, pairs=pairs, distances=distances)

        self.assertEqual(rating_graph.number_of_nodes(), 2)
        self.assertEqual(rating_graph.number_of_edges(), 1)

    def test_distance_extraction(self):
        rating_graph = MetaPathRatingGraph()
        pairs = [(1,2), (2,3), (3,5), (1,4)]
        distances = [4.2, 6, 1, 3]
        self.fill_rating_graph(graph=rating_graph, pairs=pairs, distances=distances)

        expected_distances = {
            self.meta_paths[1]: [0.0, 4.2, 10.2, 11.2, 3.0],
            self.meta_paths[2]: [0.0, 6.0, 7.0],
            self.meta_paths[3]: [0.0, 1],
            self.meta_paths[5]: [0.0],
            self.meta_paths[4]: [0.0]
        }

        for node in rating_graph.all_nodes():
            rating_graph_result = rating_graph.all_pair_distances()[rating_graph.meta_paths_map[node]]
            for distance in expected_distances[node]:
                self.assertIn(distance, list(rating_graph_result))

    def test_get_metapaths(self):
        rating_graph = MetaPathRatingGraph()
        pairs = [(0,1), (1,2), (2,3), (3,4), (4,5)]
        distances = [1, 1, 1, 1, 1]
        self.fill_rating_graph(graph=rating_graph, pairs=pairs, distances=distances)

        self.assertEqual(set(rating_graph.all_nodes()), set(self.meta_paths))

    def create_meta_path_list(self) -> List[MetaPath]:
        return [
            MetaPath(nodes=[1, 2, 1], edges=[3, 4]),
            MetaPath(nodes=[2, 5], edges=[3]),
            MetaPath(nodes=[5, 1], edges=[2]),
            MetaPath(nodes=[6, 2, 6], edges=[5, 1]),
            MetaPath(nodes=[2, 1, 1], edges=[5, 5]),
            MetaPath(nodes=[4, 1], edges=[6])
        ]

    def fill_rating_graph(self, graph, pairs, distances):
        for pair, distance in zip(pairs, distances):
            graph.add_user_rating(self.meta_paths[pair[0]], self.meta_paths[pair[1]], distance)


if __name__ == '__main__':
    unittest.main()
