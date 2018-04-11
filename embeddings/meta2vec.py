from typing import Tuple
from tensorflow import tf

class Input:
    @classmethod
    def from_json(cls, data):
        raise NotImplementedError()


    def input(self) -> Tuple:
        raise NotImplementedError()

class MetaPathsInput(Input):

    def __init__(self, dataset):
        self.dataset = self.apply_transformation(dataset)

    @classmethod
    def from_json(cls, data):
        dataset = tf.data.Dataset()

        return cls(dataset)

    def apply_transformation(self, dataset):


    @classmethod
    def parse_meta_paths(json, min_size = 5, seperator = " | "):
        walk_list = []
        available_nodes = set()
        for meta_paths in json.keys():
            node_ids = [int(id) for id in meta_paths.split(seperator)]
            if (len(node_ids) < min_size):
                continue
            walk_list.append(node_ids)
            available_nodes |= set(node_ids)
        return walk_list, available_nodes

    def input(self):
        return self.dataset