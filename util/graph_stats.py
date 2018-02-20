import pandas as pd
from util.config import ROTTEN_TOMATO_PATH
import os

class GraphStats:

    dataset_filename = 'rotten-(when_harry_met_sally-sleepless_in_seattle)-mp1-5.csv'

    def load_dataframe(self):
        df = pd.read_csv(
            os.path.join(ROTTEN_TOMATO_PATH, self.dataset_filename))
        df.columns = ['b', 'a', 'node_types', 'edge_types']
        return df

    def get_node_types(self):
        df = self.load_dataframe()

        node_types = set()
        for current_node_types_string in df.node_types:
            current_node_types = [item[0] for item in eval(current_node_types_string)]
            node_types |= set(current_node_types)

        return list(node_types)

    def get_edge_types(self):
        df = self.load_dataframe()

        edge_types = set()
        for current_edge_types_string in df.edge_types:
            current_edge_types = eval(current_edge_types_string)
            edge_types |= set(current_edge_types)

        return list(edge_types)

