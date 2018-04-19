import multiprocessing
from util.datastructures import MetaPath
from util.config import MAX_META_PATH_LENGTH, AVAILABLE_DATA_SETS, PARALLEL_EXISTANCE_PROCESSES
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
            self.import_data_set(data_set)

    @staticmethod
    def check_existence(args):
        logger = logging.getLogger('MetaExp.ExistenceCheck')
        (meta_path, data_set, edge_map, node_map) = args
        labels = []
        mp_as_list = meta_path.split("|")
        logger.debug("Checking existance of {}".format(mp_as_list))
        for i, type in enumerate(mp_as_list):
            if i % 2:
                labels.append("[n{}:{}]".format(i, edge_map[type]))
            else:
                labels.append("(e{}: {})".format(i, node_map[type]))
        logger.debug("Querying for mp {}".format(mp_as_list))
        with Neo4j(data_set['bolt-url'], data_set['username'], data_set['password']) as neo4j:
            if neo4j.test_whether_meta_path_exists("-".join(labels)):
                logger.debug("Mp {} exists!".format("-".join(labels)))
                start_node = mp_as_list[0]
                end_node = mp_as_list[-1]
                logger.debug("Adding metapath {} to record {}".format(mp_as_list, "{}_{}_{}".format(data_set['name'],
                                                                                                    start_node,
                                                                                                    end_node)))
                Redis(data_set['name'])._client.lpush("{}_{}_{}".format(data_set['name'], start_node, end_node),
                                                      pickle.dumps(MetaPath(edge_node_list=mp_as_list)))
                return mp_as_list

    def start_parallel_existence_checks(self, meta_paths: List[str], data_set: Dict) -> List[List[str]]:
        with multiprocessing.Pool(processes=PARALLEL_EXISTANCE_PROCESSES) as pool:
            args = [(mp, data_set, self.id_to_edge_type_map, self.id_to_node_type_map) for mp in meta_paths]
            return pool.map(self.check_existence, args)

    def import_data_set(self, data_set: Dict):
        self.redis = Redis(data_set['name'])
        with Neo4j(data_set['bolt-url'], data_set['username'], data_set['password']) as neo4j:
            for record in neo4j.get_meta_paths_schema(MAX_META_PATH_LENGTH):
                meta_path_list = ast.literal_eval(record['metaPaths'])
                self.logger.debug("Received list of strings {}".format(meta_path_list))
                self.logger.debug("Number of meta paths is: {}".format(len(meta_path_list)))
                self.id_to_edge_type_map = ast.literal_eval(record['edgesIDTypeDict'])
                self.id_to_node_type_map = ast.literal_eval(record['nodesIDTypeDict'])
                existing_meta_paths = self.start_parallel_existence_checks(meta_path_list, data_set)
                self.logger.debug("From {} mps {} do not exist in graph {}".format(len(meta_path_list),
                                                                                   len(existing_meta_paths),
                                                                                   data_set['name']))
                self.logger.debug(type(meta_path_list))
                self.logger.debug(type(self.id_to_edge_type_map))
                self.write_mappings(self.id_to_node_type_map, self.id_to_edge_type_map)


    def write_mappings(self, node_type_mapping: Dict[int, str], edge_type_mapping: Dict[int, str]):
        self.redis._client.hmset("{}_node_type".format(self.redis.data_set), node_type_mapping)
        self.redis._client.hmset("{}_edge_type".format(self.redis.data_set), edge_type_mapping)
        self.redis._client.hmset("{}_node_type_reverse".format(self.redis.data_set),
                                 {v: k for k, v in node_type_mapping.items()})
        self.redis._client.hmset("{}_edge_type_reverse".format(self.redis.data_set),
                                 {v: k for k, v in edge_type_mapping.items()})
