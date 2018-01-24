from explanation.explanation import Explanation
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from util.datastructures import UserOrderedMetaPaths, MetaPath


class DomainScoring():
    rated_metapaths = None
    unrated_metapaths = None

    def __init__(self, rated_metapaths: List[UserOrderedMetaPaths], unrated_metapaths: List[MetaPath]):
        """

        :param rated_metapaths: A list of lists, each inner list element represents one user weighting round.
        Each user weighting is itself a list where we care about the ordering (represents ordering on UI) and
        it contains the distance to the next element.
        """
        self.rated_metapaths = rated_metapaths
        self.unrated_metapaths = unrated_metapaths

    def score(self):
        corpus = self.rated_metapaths.extend(self.unrated_metapaths)
        vectorizer = TfidfVectorizer(analyzer='word').fit(corpus)
        # Pay attention that the argument to transform() has to be a list, otherwise sklearn iterates over the string
        X_rated = vectorizer.transform(self.rated_metapaths)
        X_unrated = vectorizer.transform(self.unrated_metapaths)
        Y_rated = self.extract_preference_learning_labels(self.rated_metapaths)
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_rated, Y_rated)
        Y_unrated = clf.predict(X_unrated)
        domain_values = self.transform_to_domain_values(Y_rated.extend(Y_unrated))
        explanation = Explanation(meta_paths=corpus, domain_value=domain_values)

    def transform_to_domain_values(self, inferred_ratings) -> List:
        """

        :param inferred_ratings: user-defined and inferred rating for all meta-paths
        :return: Total order of all meta-paths with values in [0,1]
        """
        raise NotImplementedError()
        return []

    def extract_preference_learning_labels(self) -> List:
        return []
