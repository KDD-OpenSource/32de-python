import multiprocessing
from util.datastructures import MetaPath
from util.config import MAX_META_PATH_LENGTH, AVAILABLE_DATA_SETS
from api.neo4j import Neo4j
from api.redis import Redis
from typing import Dict, List
import logging
import ast
import pickle

class RedisImporter:

    def __init__(self):
        self.redis = None
        self.logger = self.logger = logging.getLogger('MetaExp.{}'.format(__class__.__name__))
        self.id_to_edge_type_map = None
        self.id_to_node_type_map = None

    def import_all(self):
        for data_set in AVAILABLE_DATA_SETS:
            self.import_data_set(data_set['name'], data_set['bolt-url'], data_set['username'], data_set['password'])

    def check_existance(self, meta_paths: List[str], bolt_url: str, username: str, password: str) -> List[List[str]]:
            for mp in meta_paths:
                labels = []
                mp_as_list = mp.split("|")
                self.logger.debug(mp_as_list)
                self.logger.debug(self.id_to_edge_type_map)
                self.logger.debug(self.id_to_node_type_map)
                for i, type in enumerate(mp_as_list):
                    if i%2:
                        labels.append("[n{}:{}]".format(i, self.id_to_edge_type_map[type]))
                    else:
                        labels.append("(e{}: {})".format(i, self.id_to_node_type_map[type]))
                with Neo4j(bolt_url, username, password) as neo4j:
                    self.logger.debug("Querying for mp {}".format(mp_as_list))
                    if neo4j.test_whether_meta_path_exists("-".join(labels)):
                        self.logger.debug("Mp {} exists!".format("-".join(labels)))
                        yield mp_as_list


    def import_data_set(self, data_set_name: str, bolt_url: str, username: str, password: str):
        self.redis = Redis(data_set_name)
        with Neo4j(bolt_url, username, password) as neo4j:
            for record in neo4j.get_meta_paths_schema(MAX_META_PATH_LENGTH):
                meta_path_list = ast.literal_eval(record['metaPaths'])
                self.logger.debug("Received list of strings {}".format(meta_path_list))
                self.logger.debug("Number of meta paths is: {}".format(len(meta_path_list)))
                self.id_to_edge_type_map = ast.literal_eval(record['edgesIDTypeDict'])
                self.id_to_node_type_map = ast.literal_eval(record['nodesIDTypeDict'])
                existing_meta_paths = list(self.check_existance(meta_path_list, bolt_url, username, password))
                self.logger.debug("From {} mps {} do not exist in graph {}".format(len(meta_path_list),
                                                                                   len(existing_meta_paths),
                                                                                   data_set_name))
                self.logger.debug(type(meta_path_list))
                self.logger.debug(type(self.id_to_edge_type_map))
                self.write_paths(meta_path_list)
                self.write_mappings(self.id_to_node_type_map, self.id_to_edge_type_map)

    def write_paths(self, paths: List[List[str]]):
        for path in paths:
            self.write_path(path)

    def write_path(self, path: List[str]):
        start_node = path[0]
        end_node = path[-1]
        self.logger.debug("Adding metapath {} to record {}".format(path, "{}_{}_{}".format(self.redis.data_set,
                                                                                           start_node,
                                                                                           end_node)))
        self.redis._client.lpush("{}_{}_{}".format(self.redis.data_set, start_node, end_node),
                                  pickle.dumps(MetaPath(edge_node_list=path)))

    def write_mappings(self, node_type_mapping: Dict[int, str], edge_type_mapping: Dict[int, str]):
        self.redis._client.hmset("{}_node_type".format(self.redis.data_set), node_type_mapping)
        self.redis._client.hmset("{}_edge_type".format(self.redis.data_set), edge_type_mapping)
        self.redis._client.hmset("{}_node_type_reverse".format(self.redis.data_set),
                                     {v: k for k, v in node_type_mapping.items()})
        self.redis._client.hmset("{}_edge_type_reverse".format(self.redis.data_set),
                                     {v: k for k, v in edge_type_mapping.items()})