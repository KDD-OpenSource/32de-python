from typing import List, Tuple, Dict
from numpy import arange
from util.datastructures import MetaPath

SMALLER = 0
LARGER = 1

RANGE_UPPER = 1.
RANGE_LOWER = 0.

class DomainValueTransformer:
    """
    Abstract class. Domain Value Transformers are strategies that can be used to transform meta-path ordering predictions
    into an actual numeric domain value.
    """

    def transform(self, metapaths_pairs: List[Tuple[MetaPath, MetaPath]], classification: List[int]) -> List[
        Tuple[MetaPath, float]]:
        """

        :param metapaths_pairs:
        :param classification:
        :return:
        """

        raise NotImplementedError()


class NaiveTransformer(DomainValueTransformer):

    def transform(self,
                  metapaths_pairs: List[Tuple[MetaPath, MetaPath]],
                  classification: List[int],
                  all_metapaths: List[MetaPath] = None) -> List[Tuple[MetaPath, float]]:
        """
        Transforms the classified ordering of all meta-paths pairs to the domain values.
        This is done in two steps: First all the tuples are combined and flattened to a single ordered list.
        Then all meta-paths receive the value by spreading all nodes evenly between [1, 10].
        :param metapaths_pairs: All the metapath pairs that should be ordered and weighted.
        :param classification: The classificatin whether a pair is in the correct order or not.
        :param all_metapaths: A list of all meta-paths occurring in the pairs that need to be ordered.
                              If this is None it is automatically extracted.
        :return: Total order of all meta-paths with values in [0,10]
        """

        if all_metapaths is None:
            all_metapaths = self._extract_metapaths(metapaths_pairs)

        ordered_paths = self._order_pairs(dict(zip(metapaths_pairs, classification)), all_metapaths)
        return self._spread_domain_value(ordered_paths)



    def _extract_metapaths(self, metapaths_pairs: List[Tuple[MetaPath, MetaPath]]):
        all_metapaths = []

        first = metapaths_pairs[0][0]
        all_metapaths.append(first)
        for first_metapath, second_metapath in metapaths_pairs:
            if first_metapath is not first:
                break
            all_metapaths.append(second_metapath)

        return all_metapaths

    def _order_pairs(self, oracle: Dict[Tuple[MetaPath, MetaPath], int], elements: List[MetaPath]) -> List[MetaPath]:
        """
        Performs a merge sort using the oracle for comparison.
        :param oracle:
        :param all_metapaths: The
        :return:
        """

        assert elements is not None, "all_metapaths has to be specified."

        length = len(elements)
        if length < 2:
            return elements
        left = self._order_pairs(oracle, elements[:int(length/2)])
        right = self._order_pairs(oracle, elements[int(length/2):])
        return self._merge(oracle, left, right)

    def _merge(self, oracle: Dict[Tuple[MetaPath, MetaPath], int], left: List[MetaPath], right: List[MetaPath]) -> List[MetaPath]:
        """
        Merge using an oracle.
        :param oracle: The oracle tells how two given elements compare, which element is bigger or smaller.
        :param left: The left list to be merged with the right list. Left should be ordered from small to larger.
        :param right: Analog to left.
        :return: An ordered merged list of left and right based on the oracle.
        """
        ordered_paths = []
        while len(left) > 0 and len(right) > 0:
            comparer = SMALLER
            key = (left[0], right[0])
            if key not in oracle:
                comparer = LARGER
                key = (right[0], left[0])
            if oracle[key] == comparer:
                ordered_paths.append(left.pop(0))
            else:
                ordered_paths.append(right.pop(0))
        for element in left + right:
            ordered_paths.append(element)
        return ordered_paths



    def _spread_domain_value(self, ordered_paths: List[MetaPath]) -> List[Tuple[MetaPath, float]]:
        weights = arange(RANGE_UPPER, RANGE_LOWER, -RANGE_UPPER/len(ordered_paths))[::-1]
        return list(zip(ordered_paths, weights))



