import unittest
from domain_scoring.domain_scoring import DomainScoring

class MyTestCase(unittest.TestCase):
    def test_all_pairs(self):
        list = [1, 2, 3]
        all_pairs = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

        ds = DomainScoring()

        self.assertEqual(all_pairs, ds._all_pairs(list))


if __name__ == '__main__':
    unittest.main()
