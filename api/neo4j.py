from neo4j.v1 import GraphDatabase, Node
from typing import List
import logging

logger = logging.getLogger('MetaExp.Neo4j')


class Neo4j:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start_precomputation(self, mode: str, length: int):
        """
        Precomputes all meta-paths (or all meta-paths for high degree nodes, depending on 'mode') up to a
        meta-path-length given by 'length' and saves them in a file named 'Precomputed_MetaPaths.txt'
        :param mode:
        :param length:
        :return:
        """
        # mode = {full, high-degree}
        if True:
            return
        with self._driver.session() as session:
            probably_json = session.run(
                "Call algo.computeAllMetaPaths($length);",
                length=str(length))
            return probably_json.records()

    def get_metapaths(self, nodeset_A: List[int], nodeset_B: List[int], length: int):
        """
        Computes all meta-paths up to a meta-path-length given by 'length' that start with start-nodes and end with
        end-nodes given by 'nodeset_A' and 'nodeset_B' and saves them in a file named
        'Precomputed_MetaPaths_Instances.txt'
        :param nodeset_A:
        :param nodeset_B:
        :param length:
        :return:
        """
        nodeset_A = self._convert_node_set(nodeset_A)
        nodeset_B = self._convert_node_set(nodeset_B)

        with self._driver.session() as session:
            probably_json = session.run(
                "Call algo.computeAllMetaPathsForInstances($startNodeIds, $endNodeIds, $length);",
                startNodeIds=nodeset_A, endNodeIds=nodeset_B, length=str(length))
            return probably_json.records()

    def get_all_metapaths(self):
        """
        Reads and returns computed meta-paths and their counts from a file given by 'filePath'
        :return:
        """
        with self._driver.session() as session:
            probably_json = session.run(
                "Call algo.readPrecomputedMetaPaths($filePath);",
                filePath="../../../precomputed/Precomputed_MetaPaths_BioL6.txt")
            return probably_json.single()

    def get_id_label_mapping(self):
        """
        Returns the mapping from label IDs to label names
        :return:
        """
        with self._driver.session() as session:
            probably_json = session.run(
                "CALL algo.getLabelIdToLabelNameMapping();")
            return probably_json.single()

    # TODO: Implement different return types (node instances, node types)
    # TODO: What are the parameters of the neo4j procedure?
    def random_walk(self, maybe_start_id: int, maybe_number_of_random_walks: int, maybe_walk_length: int):
        with self._driver.session() as session:
            # TODO: Is the result returned as paths?
            maybe_paths = session.run(
                "CALL random_walk($maybe_start_id, $maybe_number_of_random_walks, $maybe_walk_length);",
                maybe_start_id=maybe_start_id, maybe_number_of_random_walks=maybe_number_of_random_walks,
                maybe_walk_length=maybe_walk_length)
            return maybe_paths.records()

    @staticmethod
    def _convert_node_set(nodeset: List[int]) -> str:
        str = "{"
        for id in nodeset:
            str += "{}, ".format(id)
        str = str[:-2]
        str += "}"
        return str
