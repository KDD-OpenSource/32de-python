from tokenize import String, Number
from util.datastructures import MetaPath
from util.config import MAX_META_PATH_LENGTH
from api.neo4j import Neo4j
from api.redis import Redis
from typing import Dict, List
from api.server import available_datasets
import logging


class RedisImporter:

    def __init__(self):
        self.redis = Redis()
        self.logger = self.logger = logging.getLogger('MetaExp.{}'.format(__class__.__name__))

    def import_all(self):
        for data_set in available_datasets:
            self.import_data_set(data_set['name'], data_set['bolt-url'], data_set['username'], data_set['password'])

    def import_data_set(self, data_set_name:str, bolt_url:str, username:str, password:str):
        with Neo4j(bolt_url, username, password) as  neo4j:
            for record in neo4j.get_meta_paths_schema(MAX_META_PATH_LENGTH):
                self.logger.debug(record)
                self.logger.debug(type(record['metaPaths']))
                self.logger.debug(type(record['edgesIDTypeDict']))
                self.write_paths(data_set_name, record['metaPaths'])
                self.write_mappings(data_set_name, record['nodesIDTypeDict'], record['edgesIDTypeDict'])

    def write_paths(self, data_set_name:str, paths: List[str]):
            for path in paths:
                self.write_path(data_set_name, path)

    #todo take string as index not number
    def write_path(self, data_set_name:str, path: str):
        pathAsList = path.split("|")
        startNode = pathAsList[0]
        endNode = pathAsList[-1]
        self.logger.debug("Adding metapath {} to record {}".format(path, "{}_{}_{}".format(data_set_name, startNode, endNode)))
        self.redis._client.append("{}_{}_{}".format(data_set_name, startNode, endNode), MetaPath(edge_node_list=pathAsList))

    def write_mappings(self, data_set_name:str, node_type_mapping: Dict[int, str], edge_type_mapping: Dict[int, str]):
            self.write_mapping("{}_node_type".format(data_set_name), node_type_mapping)
            self.write_mapping("{}_edge_type".format(data_set_name), edge_type_mapping)

    def write_mapping(self, key_name:str, mapping: Dict[int, str]):
        key = "{}_mapping".format(key_name)
        for field in mapping.keys():
            self.redis._client.hset(key, field, mapping[field])
            self.redis._client.hset("reverse_{}".format(key), field, mapping[field])

