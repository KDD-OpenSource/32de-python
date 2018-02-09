import unittest
from util.lists import *

class ListsTest(unittest.TestCase):


    def test_all_pairs(self):
        elements = [1, 2, 3]
        all_pairs_elem = [(1, 2), (1, 3), (2, 3)]
        all_pairs_elem_inverse = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]


        self.assertEqual(all_pairs_elem, all_pairs(elements))
        self.assertEqual(all_pairs_elem_inverse, all_pairs(elements, True))


if __name__ == '__main__':
    unittest.main()
