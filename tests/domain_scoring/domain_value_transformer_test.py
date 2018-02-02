import unittest
from domain_scoring.domain_value_transformer import NaiveTransformer, SMALLER
from util.lists import all_pairs


class NaiveDomainValueTransformerTest(unittest.TestCase):

    def setUp(self):
        self.transformer = NaiveTransformer()
        self.oracle = self._create_oracle(range(6))

    def test_transform(self):
        self.assertEqual([(1, 5), (2, 10)], self.transformer.transform([(1, 2)], [SMALLER]))

    def test_spread_domain_value(self):
        self.assertEqual(
            [(1, 5), (2, 10)],
            self.transformer._spread_domain_value([1, 2]))
        self.assertEqual(
            [(1, 2.5), (2, 5), (3, 7.5), (4, 10)],
            self.transformer._spread_domain_value([1, 2, 3, 4]))
        self.assertEqual(
            [(1, 5), (2, 10)],
            self.transformer._spread_domain_value([1, 2]))
        self.assertEqual(
            [(1, 10)],
            self.transformer._spread_domain_value([1]))

    def test_order_pairs(self):
        self.assertEqual([0, 1, 2, 3, 4, 5], self.transformer._order_pairs(self.oracle, [5, 3, 1, 0, 2, 4]))
        self.assertEqual([1, 2, 3, 4, 5], self.transformer._order_pairs(self.oracle, [5, 3, 1, 2, 4]))

    def test_merge(self):
        self.assertEqual([1, 2, 3, 4, 5], self.transformer._merge(self.oracle, [1, 2, 3], [4, 5]))
        self.assertEqual([1, 2, 3], self.transformer._merge(self.oracle, [3], [1, 2]))
        self.assertEqual([0, 1, 2, 3, 4, 5], self.transformer._merge(self.oracle, [1, 4, 5], [0, 2, 3]))
        self.assertEqual([0, 1, 2, 3, 4, 5], self.transformer._merge(self.oracle, [], [0, 1, 2, 3, 4, 5]))
        self.assertEqual([0, 1, 2, 3, 4, 5], self.transformer._merge(self.oracle, [0, 1, 2, 3, 4, 5], []))

    def test_extract_metapaths(self):
        pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

        self.assertEqual([1, 2, 3, 4], self.transformer._extract_metapaths(pairs))

    def _create_oracle(self, elements):
        all_elements = all_pairs(elements)
        return dict(zip(all_elements, [SMALLER] * len(all_elements)))

if __name__ == '__main__':
    unittest.main()
