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

def constant(mp: MetaPath) -> float:
    """
    Always rate the paths the same.
    """
    return 1.0

def cmd_line(mp: MetaPath) -> float:
    print("Please rate this meta-path: {}".format(mp))
    rating = input()
    return float(rating)