from tokenize import String, Number
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
        self.redis = Redis()
        self.logger = self.logger = logging.getLogger('MetaExp.{}'.format(__class__.__name__))

    def import_all(self):
        for data_set in AVAILABLE_DATA_SETS:
            self.import_data_set(data_set['name'], data_set['bolt-url'], data_set['username'], data_set['password'])

    def import_data_set(self, data_set_name: str, bolt_url: str, username: str, password: str):
        with Neo4j(bolt_url, username, password) as neo4j:
            for record in neo4j.get_meta_paths_schema(MAX_META_PATH_LENGTH):
                self.logger.debug(record)
                meta_path_list = ast.literal_eval(record['metaPaths'])
                id_to_edge_type_dict = ast.literal_eval(record['edgesIDTypeDict'])
                id_to_node_type_dict = ast.literal_eval(record['nodesIDTypeDict'])
                self.logger.debug(type(meta_path_list))
                self.logger.debug(type(id_to_edge_type_dict))
                self.write_paths(data_set_name, meta_path_list)
                self.write_mappings(data_set_name, id_to_node_type_dict, id_to_edge_type_dict)

    def write_paths(self, data_set_name: str, paths: List[str]):
        self.logger.debug("Received list of strings {}".format(paths))
        self.logger.debug("Number of meta paths is: {}".format(len(paths)))
        for path in paths:
            self.write_path(data_set_name, path)

    # Todo take string as index not number
    def write_path(self, data_set_name: str, path: str):
        path_as_list = path.split("|")
        start_node = path_as_list[0]
        end_node = path_as_list[-1]
        self.logger.debug("Adding metapath {} to record {}".format(path, "{}_{}_{}".format(data_set_name, start_node, end_node)))
        self.redis._client.lpush("{}_{}_{}".format(data_set_name, start_node, end_node),
                                  pickle.dumps(MetaPath(edge_node_list=path_as_list)))

    def write_mappings(self, data_set_name: str, node_type_mapping: Dict[int, str], edge_type_mapping: Dict[int, str]):
            self.write_mapping("{}_node_type".format(data_set_name), node_type_mapping)
            self.write_mapping("{}_edge_type".format(data_set_name), edge_type_mapping)

    def write_mapping(self, key_name: str, mapping: Dict[int, str]):
        key = "{}_mapping".format(key_name)
        self.redis._client.hmset(key, mapping)
