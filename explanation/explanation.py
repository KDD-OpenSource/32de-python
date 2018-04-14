from typing import List
from util.datastructures import MetaPathRating
from util.config import BASELINE_MODE

import numpy as np


class Explanation:
    """
    Computes similar nodes based on both node sets and their occurring meta-paths with belonging
    domain and structural value.
    """

    @staticmethod
    def get_similar_nodes():
        """
        :return: Array of dictionaries, that hold a 1-neighborhood query and properties about
                 k-similar nodes regarding both node sets
        TODO: Return similar nodes based on graph embeddings. Build 1-neighborhood query and read property of node
              dynamically based on Node-ID
        """
        similar_nodes = [
            {
                "cypher_query": "MATCH (n) RETURN n LIMIT 1",
                "properties": {
                    "name": "Node A",
                    "label": "Node Type A"
                }
            },
            {
                "cypher_query": "MATCH (n) RETURN n LIMIT 1",
                "properties": {
                    "name": "Node B",
                    "label": "Node Type B"
                }
            },
            {
                "cypher_query": "MATCH (n) RETURN n LIMIT 1",
                "properties": {
                    "name": "Node C",
                    "label": "Node Type A"
                }
            },
            {
                "name": "Node D",
                "cypher_query": "MATCH (n) RETURN n LIMIT 1",
                "properties": {
                    "name": "Node D",
                    "label": "Node Type B"
                }
            }
        ]

        return similar_nodes


class SimilarityScore:
    """
    Computes similarity score between the two node sets.
    Computes contribution of each meta-path to overall similarity score
    """

    meta_paths = None
    similarity_score = None
    algorithm_type = None

    def __init__(self, algorithm_type=BASELINE_MODE):
        self.algorithm_type = algorithm_type

    def fetch_meta_paths(self) -> List[MetaPathRating]:
        """
        :return: List of meta-path objects between both node sets
        TODO: Fetch meta-paths dynamically. At the moment, we pass meta paths directly from the tests.
        """

        return self.meta_paths

    def calculate_similarity(self, meta_path_ratings: List[MetaPathRating]) -> float:
        """
        Computes a sum of a linear combination of structural and domain value
        over all meta-paths, normalized by each meta-path length. First simplified,
        not experimentally tested baseline.
        :return: similarity score between both node sets as float
        """

        structural_values = np.array([])
        domain_values = np.array([])

        for meta_path_rating in meta_path_ratings:
            structural_values = np.append(structural_values, [meta_path_rating.structural_value])
            domain_values = np.append(domain_values, [meta_path_rating.domain_value])

        self.similarity_score = np.sum(structural_values * domain_values) / len(meta_path_ratings)
        return self.similarity_score

    @staticmethod
    def get_similarity_score() -> float:
        """
        :return: similarity score between both node sets as float
        TODO: Return similarity score dynamically after computation
        """

        return 10.53

    @staticmethod
    def get_contributing_meta_path(meta_path_id: int):
        """
        :param meta_path_id: Integer, that is a unique identifier for a meta-path
        :return: Dictionary, that holds detailed information about the belonging meta-path
        TODO: Take structural value depending on given meta_path. Compute contribution information dynamically
        """
        contributing_meta_path = {
            "id": meta_path_id,
            "name": "Meta-Path " + str(meta_path_id),
            "structural_value": 10,
            "contribution_ranking": 2,
            "contribution_value": 25.55,
            "meta_path": "PERSON - acted_id - MOVIE - directed - PERSON - directed - MOVIE",
            "instance_queries": [
                "MATCH (n)-[r]->(m) RETURN n,r,m",
                "MATCH (n)-[r]->(m) RETURN n,r,m",
                "MATCH (n)-[r]->(m) RETURN n,r,m"
            ]
        }

        return contributing_meta_path

    @staticmethod
    def get_contributing_meta_paths():
        """
        :return: List of dictionaries, that hold necessary information for a pie chart visualization
                 about k-most contributing meta-paths to overall similarity score
        TODO: Compute values of k-most contributing meta-paths dynamically
        """
        contributing_meta_paths = [
            {
                "id": 1,
                "label": "make",
                "value": 551,
                "color": "hsl(131, 70%, 50%)"
            },
            {
                "id": 2,
                "label": "erlang",
                "value": 226,
                "color": "hsl(358, 70%, 50%)"
            },
            {
                "id": 3,
                "label": "c",
                "value": 129,
                "color": "hsl(151, 70%, 50%)"
            },
            {
                "id": 4,
                "label": "php",
                "value": 67,
                "color": "hsl(52, 70%, 50%)"
            },
            {
                "id": 5,
                "label": "java",
                "value": 452,
                "color": "hsl(221, 70%, 50%)"
            },
            {
                "id": 6,
                "label": "stylus",
                "value": 406,
                "color": "hsl(102, 70%, 50%)"
            },
            {
                "id": 7,
                "label": "ruby",
                "value": 433,
                "color": "hsl(341, 70%, 50%)"
            }
        ]

        return contributing_meta_paths
