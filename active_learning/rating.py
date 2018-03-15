from util.datastructures import MetaPath
import numpy as np

"""
A collection of functions for rating a domain-based score of a meta-path.
"""


def length_based(mp: MetaPath) -> float:
    """
    Set the rating as the length of the metapath.
    """
    return float(len(mp))

def randomly(mp: MetaPath) -> float:
    """
    Rate the metapath at a random valuein [0,1].
    """
    return np.random.rand()