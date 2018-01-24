from explanation.explanation import Explanation
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier


class DomainScoring():
    def __init__(self, rated_metapaths: List):
        """

        :param rated_metapaths: A list of lists, each inner list element represents one user weighting round.
        Each user weighting is itself a list where we care about the ordering (represents ordering on UI) and
        it contains the distance to the next element.
        """
        raise NotImplementedError()

    def score(self):
        rated_metapaths = ["SNP IN POSITIONWINDOW NEXT POSITIONWINDOW IN LOCUS POS TRANSCRIPT",
                           "SNP IN POSITIONWINDOW IN LOCUS POS TRANSCRIPT"]
        unrated_metapaths = []
        corpus = rated_metapaths.extend(unrated_metapaths)
        vectorizer = TfidfVectorizer(analyzer='word').fit(corpus)
        # Pay attention that the argument to transform() has to be a list, otherwise sklearn iterates over the string
        X_rated = vectorizer.transform(rated_metapaths)
        X_unrated = vectorizer.transform(unrated_metapaths)
        # TODO: What is Y?
        Y_rated = []
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_rated, Y_rated)
        Y_unrated = clf.predict(X_unrated)
        domain_values = self.transform_to_domain_values(Y_rated.extend(Y_unrated))
        explanation = Explanation(X=corpus, Y=domain_values)

    def transform_to_domain_values(self, inferred_ratings) -> List:
        """

        :param inferred_ratings: user-defined and inferred rating for all meta-paths
        :return: Total order of all meta-paths with values in [0,1]
        """
        raise NotImplementedError()
        return []
