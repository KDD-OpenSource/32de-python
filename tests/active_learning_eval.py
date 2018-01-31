import os
import unittest

import pandas as pd

from active_learning.active_learning import *
from active_learning.oracles import MockOracle
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

    def _eval_complete_dataset_on(self,oracle,meta_paths):
        rating = MetaPathRatingGraph()
        ratings = map(oracle.rate_meta_path, meta_paths)
        ordered = sorted(zip(ratings, meta_paths), key=lambda tuple: tuple[0])
        list(map(lambda x, y: rating.add_user_rating(x[1], y[1], y[0] - x[0]), ordered[:-1], ordered[1:]))
        return rating

    def test_eval_dataset(self):
        # TODO: parametrize for several different oracles, learning algorithms
        active_learner = ActiveLearner(oracle=MockOracle(), algorithm=ITERATIVE_BATCHING, data_set=self.data_set)
        achieved_user_rating = active_learner.retrieve_rating()
        complete_user_rating = self._eval_complete_dataset_on(MockOracle(),self.data_set)
        print(achieved_user_rating)
        print(complete_user_rating)
        achieved_user_rating.draw(filename='log/achieved_rating.png')
        complete_user_rating.draw(filename='log/complete_rating.png')

        # TODO compare achieved and compelete?
        # method one: precision
        # since the model does not lie (we use the same instance)
        # we will never get false positives (might change in real-world-scenerio...)
        # so how can we evaluate? => only in combination with preference leared from our suggestion!

        # random oracle for relative scoring?

        # statistical methods/ aggregation
        # min/max values and outlier => are the interest 'groupings' similar

        # after preference learning complete comparison
        # MSE (relative?)
        # ranking similarity

        #

if __name__ == '__main__':
    unittest.main()
