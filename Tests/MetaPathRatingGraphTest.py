import unittest
from util.datastructures import MetaPathRatingGraph, MetaPath

class MetaPathRatingTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(MetaPathRatingTest, self).__init__(*args, **kwargs)
        self.rating_graph = MetaPathRatingGraph()

    def test_creation(self):
        self.assertTrue(self.rating_graph is not None)

    def fill_rating_graph(self):
        for i in range(10)
        MetaPath(nodes=[], edges=[])
        self.rating_graph.add_user_rating()

if __name__ == '__main__':
    unittest.main()
