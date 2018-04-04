from util.datastructures import MetaPath

"""
A collection of functions for rating a domain-based score of a meta-path.
"""


def length_based(mp: MetaPath) -> float:
    """
    Set the rating as the length of the metapath.
    """
    return float(len(mp))
