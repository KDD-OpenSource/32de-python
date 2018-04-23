import redis
import logging
import pickle

from util.datastructures import MetaPath
from typing import List, Tuple
from util.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD


class Redis:

    def __init__(self, data_set_name: str):
        self._client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASSWORD)
        self.data_set = data_set_name
        self.logger = logging.getLogger('MetaExp.{}'.format(__class__.__name__))

    def meta_paths(self, start_type: str, end_type: str) -> List:
        self.logger.debug("Retrieving meta paths...")
        pickled_list = self._client.lrange("{}_{}_{}_embedded".format(self.data_set, start_type, end_type), 0, -1)
        self.logger.debug("Number of meta paths for {} and {} is {}".format(start_type, end_type, len(pickled_list)))
        return [pickle.loads(pickled_entry) for pickled_entry in pickled_list]

    def id_to_edge_type_map(self):
        return self._client.hgetall("{}_edge_type_map".format(self.data_set))

    def id_to_node_type_map(self):
        return self._client.hgetall("{}_node_type_map".format(self.data_set))

    def node_type_to_id_map(self):
        return self._client.hgetall("{}_node_type_map_reverse".format(self.data_set))

    def edge_type_to_id_map(self):
        return self._client.hgetall("{}_edge_type_map_reverse".format(self.data_set))

    def get_all_meta_paths(self):
        result = []
        # TODO: Match only keys without '_embedding'
        for key in self._client.keys(pattern='{}_[0-9-]*_[0-9-]*'.format(self.data_set)):
            result.extend([[int(type) for type in pickle.loads(pickled_entry).get_representation('UI')]
                           for pickled_entry in self._client.lrange(key, 0, -1)])
        return result

    def store_embeddings(self, mp_embeddings_list: List[Tuple[List[str], List[float]]]):
        for mp, embedding in mp_embeddings_list:
            self.logger.debug("Got mp {}".format(mp))
            node_type_map, edge_type_map = self.id_to_node_type_map(), self.id_to_edge_type_map()
            start_type, end_type = node_type_map[str(mp[0]).encode()].decode(), node_type_map[str(mp[-1]).encode()].decode()
            node_list = [node_type_map[str(node).encode()].decode() for node in mp[::2]]
            edge_list = [edge_type_map[str(edge).encode()].decode() for edge in mp[1::2]]
            meta_path = MetaPath(nodes=node_list, edges=edge_list)
            meta_path.store_embedding(embedding)
            self.logger.debug("Created meta path object {}".format(meta_path))
            self._client.lpush("{}_{}_{}_embedded".format(self.data_set, start_type, end_type), pickle.dumps(meta_path))
