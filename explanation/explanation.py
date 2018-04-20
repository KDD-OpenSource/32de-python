from typing import List
from util.datastructures import MetaPathRating
from util.config import BASELINE_MODE
import numpy as np
from api.neo4j_own import Neo4j
import logging


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
    meta_paths_top_k = None
    similarity_score = None
    algorithm_type = None
    sum_structural_values = 0
    get_complete_rating = None
    dataset = None
    start_node_ids = []
    end_node_ids = []

    def __init__(self, get_complete_rating, dataset, start_node_ids, end_node_ids, algorithm_type=BASELINE_MODE):
        self.algorithm_type = algorithm_type
        self.get_complete_rating = get_complete_rating
        self.dataset = dataset
        self.start_node_ids = start_node_ids
        self.end_node_ids = end_node_ids
        self.logger = logging.getLogger('MetaExp.{}'.format(self.__class__.__name__))

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes.
        d = dict(self.__dict__)
        # Remove the unpicklable entries.
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.logger = logging.getLogger('MetaExp.{}'.format(__class__.__name__))

    def refresh(self):
        self.meta_paths_top_k = []
        self.meta_paths = self.get_complete_rating()
        self.logger.debug(self.meta_paths)
        domain_values = np.array([mp['domain_value'] for mp in self.meta_paths])
        top_k_domain_idx = self.apply_low_pass_filtering(domain_values, 20)

        for i in top_k_domain_idx:
            self.meta_paths_top_k.append(self.meta_paths[i])

        with Neo4j(uri=self.dataset['bolt-url'], user=self.dataset['username'],
                   password=self.dataset['password']) as neo4j:
            for mp in self.meta_paths_top_k:
                self.logger.debug(type(neo4j.get_structural_value(mp.meta_path, self.start_node_ids, self.end_node_ids)))
                # self.logger.debug(neo4j.get_structural_value(mp.meta_path, self.start_node_ids, self.end_node_ids))

        return True

    def calculate_similarity(self, meta_path_ratings: List[MetaPathRating],
                             low_pass_filter=False, filter_rate=100, use_soft_max=False,
                             normalize_structural_values=True) -> float:
        """
        Computes a sum of a linear combination of structural and domain value
        over all meta-paths. First simplified, not experimentally tested baseline.
        :return: similarity score between both node sets as float
        """
        structural_values = np.array([meta_path_rating.structural_value for meta_path_rating in meta_path_ratings])
        domain_values = np.array([meta_path_rating.domain_value for meta_path_rating in meta_path_ratings])

        self.sum_structural_values = np.sum(structural_values)

        if use_soft_max:
            structural_values = self.apply_soft_max(structural_values)

        if low_pass_filter:
            structural_values = self.apply_low_pass_filtering(structural_values, filter_rate)

        if normalize_structural_values:
            structural_values = np.vectorize(self.get_normalized_structural_value)

        normalized_structural_values = structural_values(structural_values)
        self.similarity_score = np.sum(normalized_structural_values * domain_values)
        return self.similarity_score

    @staticmethod
    def apply_soft_max(input_array: List[float]) -> List[float]:
        return np.exp(input_array) / np.sum(np.exp(input_array))

    @staticmethod
    def apply_low_pass_filtering(input_array: List[float], filter_rate: int) -> List[float]:
        return np.argsort(input_array)[-filter_rate:]

    def get_normalized_structural_value(self, structural_value: float) -> float:
        return structural_value / self.sum_structural_values

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
            "instance_query": "MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 2"
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
