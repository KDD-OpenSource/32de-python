from util.datastructures import MetaPath
from util.config import MAX_META_PATH_LENGTH, AVAILABLE_DATA_SETS
from api.neo4j_own import Neo4j
from api.redis_own import Redis
from typing import Dict, List, Tuple
import logging
import ast
import pickle
import embeddings

class RedisImporter:

    def __init__(self):
        self.redis = None
        self.logger = self.logger = logging.getLogger('MetaExp.{}'.format(__class__.__name__))

    def import_all(self):
        for data_set in AVAILABLE_DATA_SETS:
            self.import_data_set(data_set['name'], data_set['bolt-url'], data_set['username'], data_set['password'])

    def import_data_set(self, data_set_name: str, bolt_url: str, username: str, password: str):
        self.redis = Redis(data_set_name)
        with Neo4j(bolt_url, username, password) as neo4j:
            for record in neo4j.get_meta_paths_schema(MAX_META_PATH_LENGTH):
                meta_path_list = ast.literal_eval(record['metaPaths'])
                self.logger.debug("Received list of strings {}".format(meta_path_list))
                self.logger.debug("Number of meta paths is: {}".format(len(meta_path_list)))
                meta_path_embedded_original = embeddings.meta2vec.calculate_embeddings(
                    [mp.split("|") for mp in meta_path_list])
                id_to_edge_type_dict = ast.literal_eval(record['edgesIDTypeDict'])
                id_to_node_type_dict = ast.literal_eval(record['nodesIDTypeDict'])
                self.logger.debug(type(meta_path_list))
                self.logger.debug(type(id_to_edge_type_dict))
                self.write_paths(meta_path_embedded_original)
                self.write_mappings(id_to_node_type_dict, id_to_edge_type_dict)

    def write_paths(self, paths_original_embedded: List[Tuple(List[str], List[float])]):
        for path_original, path_embedded in paths_original_embedded:
            self.write_path(path_original, path_embedded)

    def write_path(self, path_original: List[str], path_embedded: List[float]):
        start_node = path_original[0]
        end_node = path_original[-1]
        self.logger.debug("Adding metapath {} to record {}".format(path_original, "{}_{}_{}".format(self.redis.data_set,
                                                                                                    start_node,
                                                                                                    end_node)))
        # Todo add embedding to MetaPath object.
        self.redis._client.lpush("{}_{}_{}".format(self.redis.data_set, start_node, end_node),
                                  pickle.dumps(MetaPath(edge_node_list=path_original)))
        self.redis._client.lpush("{}_{}_{}_embedded".format(self.redis.data_set, start_node, end_node),
                                  pickle.dumps(path_embedded))

    def write_mappings(self, node_type_mapping: Dict[int, str], edge_type_mapping: Dict[int, str]):
        self.redis._client.hmset("{}_node_type".format(self.redis.data_set), node_type_mapping)
        self.redis._client.hmset("{}_edge_type".format(self.redis.data_set), edge_type_mapping)
        self.redis._client.hmset("{}_node_type_reverse".format(self.redis.data_set),
                                 {v: k for k,v in node_type_mapping.items()})
        self.redis._client.hmset("{}_edge_type_reverse".format(self.redis.data_set),
                                 {v: k for k,v in edge_type_mapping.items()})

