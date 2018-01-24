from explanation.explanation import Explanation
from typing import List
import util.config as config


class DomainScoring():
    def __init__(self, weighted_metapaths: List, mode: str = config.RESEARCH_MODE):
        """

        :param weighted_metapaths: A list of lists, each inner list element represents one user weighting round.
        Each user weighting is itself a list where we care about the ordering (represents ordering on UI) and
        it contains the distance to the next element.
        """
        raise NotImplementedError()

    def score(self):
        explanation = Explanation()
