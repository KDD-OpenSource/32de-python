from typing import List, Tuple

def all_pairs(elements: List, inverse: bool = False) -> List[Tuple]:
    """

    :param elements: the list from which the elements are taken.
    :param inverse: if True also the inverse pairs are added.
    :return: All pairs found in the input list.
    """

    pairs = []
    for key in range(len(elements)):
        element = elements[key]
        lower = 0 if inverse else key
        for successor in elements[lower:]:
            if element is successor:
                continue

            pairs.append((element, successor))
    return pairs