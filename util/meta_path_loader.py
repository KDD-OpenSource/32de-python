from abc import ABC, abstractmethod
from .datastructures import MetaPath
from typing import List
import pandas as pd
from .config import ROTTEN_TOMATO_PATH
import os
import io


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
        return MetaPath(nodes=nodes, edges=edges)


class RottenTomatoMetaPathLoader(AbstractMetaPathLoader):
    dataset_filename = 'rotten-(when_harry_met_sally-sleepless_in_seattle)-mp1-5.csv'

    def load_meta_paths(self) -> List[MetaPath]:
        df = pd.read_csv(
            os.path.join(ROTTEN_TOMATO_PATH, self.dataset_filename))
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
        print("{}: Number of meta-paths is {}".format(self.__class__.__name__.upper(), len(meta_paths)))
        return meta_paths


class CypherDataSetLoader(AbstractMetaPathLoader):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_meta_paths(self) -> List[MetaPath]:
        with open(self.dataset_path, 'r') as f:
            lines = f.readlines()
            fo = io.StringIO()
            new_lines = [lines[0].replace(',', ';')]
            for line in lines[1:]:
                seps = line.split(',')
                line = ';'.join(seps[:2])
                brack = ','.join(seps[2:-1])
                line += ';' + ']];['.join(brack.split(']], [')) + ';' + seps[-1]
                new_lines.append(line)
            fo.writelines(u"" + line for line in new_lines)
            fo.seek(0)

        df = pd.read_csv(fo, index_col=None, sep=';')

        df.columns = [col_name.strip() for col_name in df.columns]
        df.nodes_types = df.nodes_types.apply(eval)
        df.relationship_types = df.relationship_types.apply(eval)
        meta_paths = []
        for i, row in df.iterrows():
            nodes = [i[0] for i in row.nodes_types]
            edges = [rel_type.split('/')[-1] for rel_type in row.relationship_types]
            meta_paths.append(MetaPath(nodes=nodes, edges=edges))
        print("{}: Number of meta-paths is {}".format(self.__class__.__name__.upper(), len(meta_paths)))
        return meta_paths

class CypherDataSetLoaderWithoutCounts(AbstractMetaPathLoader):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_meta_paths(self) -> List[MetaPath]:
        with open(self.dataset_path, 'r') as f:
            lines = f.readlines()
            fo = io.StringIO()
            new_lines = [lines[0].replace(',', ';')]
            for line in lines[1:]:
                seps = line.split(',')
                line = ';'.join(seps[:2])
                brack = ','.join(seps[2:])
                line += ';' + ']];['.join(brack.split(']], ['))
                new_lines.append(line)
            fo.writelines(u"" + line for line in new_lines)
            fo.seek(0)

        df = pd.read_csv(fo, index_col=None, sep=';')

        df.columns = [col_name.strip() for col_name in df.columns]
        df.drop_duplicates(inplace=True, subset=['nodes_types', 'relationship_types'])
        df.nodes_types = df.nodes_types.apply(eval)
        df.relationship_types = df.relationship_types.apply(eval)
        meta_paths = []
        for i, row in df.iterrows():
            nodes = [i[0] for i in row.nodes_types]
            edges = [rel_type.split('/')[-1] for rel_type in row.relationship_types]
            meta_paths.append(MetaPath(nodes=nodes, edges=edges))
        print("{}: Number of meta-paths is {}".format(self.__class__.__name__.upper(), len(meta_paths)))
        return meta_paths