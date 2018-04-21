from neo4j.v1 import GraphDatabase
from typing import List
from util.datastructures import MetaPath
import logging
from api.redis_own import Redis


class Neo4j:
    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logging.getLogger('MetaExp.{}'.format(__class__.__name__))

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start_precomputation(self, mode: str, length: int, ratio: float = 0.0):
        """
        Precomputes all meta-paths (or all meta-paths for high degree nodes, depending on 'mode') up to a
        meta-path-length given by 'length' and saves them in a file named 'Precomputed_MetaPaths.txt'
        :param ratio:
        :param mode:
        :param length:
        :return:
        """
        if mode == "full":
            with self._driver.session() as session:
                probably_json = session.run(
                    "Call algo.computeAllMetaPaths($length);",
                    length=str(length))
                return probably_json.records()
        else:
            with self._driver.session() as session:
                probably_json = session.run(
                    "Call algo.metaPathPrecomputeHighDegreeNodes($length, $ratio);",
                    length=str(length), ratio=str(ratio))
                return probably_json.records()

    def sleep(self, time: int):
        """

        :param time:
        :return:
        """
        with self._driver.session() as session:
            # apoc procedure expects int as argument.
            probably_json = session.run(
                "Call apoc.util.sleep($duration);", duration=time)
            return probably_json.records()

    def get_meta_paths_for_node_types(self, startType: str, endType: str, length: int):
        """

        :param startType:
        :param endType:
        :param length:
        :return:
        """
        with self._driver.session() as session:
            json = session.run("Call algo.computeMetaPathsBetweenTypes($length, $startType, $endType);",
                               length=str(length), startType=startType, endType=endType)
            records = json.records()
            self.logger.debug(records)
            return records

    def get_structural_value(self, meta_path: MetaPath, start_nodes: List, end_nodes: List, dataset_name: str):
        redis = Redis(dataset_name)
        id_to_node_type_map = redis.id_to_node_type_map()
        id_to_edge_type_map = redis.id_to_edge_type_map()
        mp = meta_path.as_list()
        nodes = ['(n{}:{})'.format(i, id_to_node_type_map[n.encode()].decode()) for i, n in enumerate(mp[::2])]
        n = len(nodes)
        edges = ['[e{}:{}]'.format(i, id_to_edge_type_map[e.encode()].decode()) for i, e in enumerate(mp[1::2])]
        all = nodes + edges
        all[::2] = nodes
        all[1::2] = edges
        path = '-'.join(all)
        start_ids = '[' + ','.join(map(str, start_nodes)) + ']'
        end_ids = '[' + ','.join(map(str, end_nodes)) + ']'
        query = "MATCH p = {} " \
                "WHERE ID(n0) in {} and ID(n{}) in {} " \
                "RETURN count(*) as count;".format(path, start_ids, n - 1, end_ids)
        self.logger.debug("Querying for '{}'".format(query))
        with self._driver.session() as session:
            statement_result = session.run(query)
            return list(statement_result.records())[0]['count']

    def test_whether_meta_path_exists(self, meta_path_query_string):
        self.logger.debug("Received match query string {}".format(meta_path_query_string))
        query = "cypher planner=rule MATCH p = {} " \
                "RETURN p limit 1".format(meta_path_query_string)
        self.logger.debug("Querying for '{}'".format(query))
        with self._driver.session() as session:
            record = session.run(query).single()
            self.logger.debug(record)
            return bool(record)

    def get_meta_paths_schema(self, length: int):
        """

        :param length:
        :return:
        """
        with self._driver.session() as session:
            statement_result = session.run("Call algo.computeAllMetaPathsSchemaFull($length);",
                                           length=str(length))
            return statement_result.records()

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
        """

        :param maybe_start_id:
        :param maybe_number_of_random_walks:
        :param maybe_walk_length:
        :return:
        """
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
