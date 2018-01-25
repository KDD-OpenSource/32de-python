import unittest
import pandas as pd
import os
from active_learning.active_learning import ActiveLearner, MockOracle
from util.datastructures import MetaPath


class ActiveLearningTest(unittest.TestCase):
    HARRY_SALLY_MATRIX = "rotten-(when_harry_met_sally-sleepless_in_seattle)-mp1-5.csv"

    test_sets = ['random', 'user_rated', 'structural']
    data_set = None

    def __init__(self, *args, **kwargs):
        super(ActiveLearningTest, self).__init__(*args, **kwargs)
        self.data_set = self._create_dataset()

    def _create_dataset(self, data_set=HARRY_SALLY_MATRIX):
        def string_to_path(node_types, edge_types):
            nodes = [item[0] for item in eval(node_types)]
            edges = eval(edge_types)
            return MetaPath(nodes, edges)

        df = pd.read_csv(
            os.path.join("data", "rotten_tomatoes", data_set))
        df.columns = ['b', 'a', 'node_types', 'edge_types']
        df.a = df.a.apply(eval)
        df.b = df.b.apply(eval)
        df_counts = df.groupby(["node_types", "edge_types"]).count().copy()
        df_counts.columns = ['structure_value', 'domain_value']
        meta_paths = []
        for i, row in df_counts.iterrows():
            nodes = i[0]
            edges = i[1]
            mp = string_to_path(nodes, edges)
            meta_paths.append(mp)
        return meta_paths

    def test_eval_dataset(self):
        # TODO: parametrize for several different oracles, learning algorithms
        active_learner = ActiveLearner(oracle=MockOracle(), dataset=self.data_set)
        user_rating = active_learner.retrieve_user_rating()

        # TODO: compare user rating to actual user rating


if __name__ == '__main__':
    unittest.main()
