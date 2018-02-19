from abc import ABC, abstractmethod
from .datastructures import MetaPath
from typing import List, Dict
import pandas as pd
from .config import ROTTEN_TOMATO_PATH
import os

class AbstractMetaPathLoader(ABC):
    """
    Abstract base object to extract meta paths from file.
    """

    @abstractmethod
    def load_meta_paths(self) -> List[MetaPath]:
        """
        :return: List of all meta paths extracted from file
        """
        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is AbstractMetaPathLoader:
            mandatory_methods = ["load_meta_paths"]
            if all(any(method in B.__dict__ for B in C.__mro__) for method in mandatory_methods):
                return True
        return NotImplemented

    def string_to_meta_path(self, node_types, edge_types):
        nodes = [item[0] for item in eval(node_types)]
        edges = eval(edge_types)
        return MetaPath(nodes, edges)


class RottenTomatoMetaPathLoader(AbstractMetaPathLoader):

    dataset_filename = 'rotten-(when_harry_met_sally-sleepless_in_seattle)-mp1-5.csv'

    def load_meta_paths(self) -> List[MetaPath]:

        df = pd.read_csv(
            os.path.join("../", ROTTEN_TOMATO_PATH, self.dataset_filename))
        df.columns = ['b', 'a', 'node_types', 'edge_types']
        df.a = df.a.apply(eval)
        df.b = df.b.apply(eval)
        df_counts = df.groupby(["node_types", "edge_types"]).count().copy()
        df_counts.columns = ['structure_value', 'domain_value']
        meta_paths = []
        for i, row in df_counts.iterrows():
            nodes = i[0]
            edges = i[1]
            mp = self.string_to_meta_path(nodes, edges)
            meta_paths.append(mp)
        print(len(meta_paths))
        return meta_paths
