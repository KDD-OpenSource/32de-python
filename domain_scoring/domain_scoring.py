from explanation.explanation import Explanation
from typing import List
from main import RESEARCH_MODE
from typing import List, Tuple
from util.ranking_graph import RankingGraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier


class DomainScoring():
    def __init__(self, rated_metapaths: List, mode: str = RESEARCH_MODE):
        """

        :param rated_metapaths: A list of lists, each inner list element represents one user weighting round.
        Each user weighting is itself a list where we care about the ordering (represents ordering on UI) and
        it contains the distance to the next element.
        """
        self.vectorizer = TfidfVectorizer(analyzer='word')


    def score(self, metapath_graph: RankingGraph, unknown: List):
        # TODO: Remove this legacy - it is kept as a reference for now.
        # rated_metapaths = [[["SNP IN POSITIONWINDOW NEXT POSITIONWINDOW IN LOCUS POS TRANSCRIPT", 0.1],
        #                     ["SNP IN POSITIONWINDOW IN LOCUS POS TRANSCRIPT", 0.2]]]
        # unrated_metapaths = [["SNP IN POSITIONWINDOW NEXT POSITIONWINDOW IN POSITIONWINDOW IN LOCUS POS TRANSCRIPT"]]
        # corpus = rated_metapaths.extend(unrated_metapaths)

        self.fit_vectorizer(metapath_graph)
        x_train, y_train = self.extract_features_labels(metapath_graph)
        x_predict = self.all_pairs(unknown)

        clf = DecisionTreeClassifier()
        clf = clf.fit(self.preprocess(x_train), y_train)

        y_predict = clf.predict(self.preprocess(x_predict))

        # TODO:
        # corpus, domain_values = self.transform_to_domain_values(Y_rated.extend(Y_unrated))
        # explanation = Explanation(meta_paths=corpus, domain_value=domain_values)

    def preprocess(self, data: List[Tuple]) -> List:
        """
        Takes a list of metapaths pairs (a, b) and vectorizes a and b and joins them to one element for training.
        :param data: the data to preprocess.
        :return: preprocessed data.
        """
        vectors = []
        for datum in data:
            vectors.append(self.vectorizer.transform(datum[0]) + self.vectorizer.transform(datum[1]))

        return vectors

    def all_pairs(self, list: List, inverse: bool = False) -> List[Tuple]:
        """

        :param list: the list from which the elements are taken.
        :param inverse: if True also the inverse pairs are added.
        :return: All pairs found in the input list.
        """

        pairs = []
        for element in list:
            for successor in list:
                pairs.append((element, successor))
                if inverse:
                    pairs.append((successor, element))
        return pairs


    def transform_to_domain_values(self, inferred_ratings) -> List:
        """

        :param inferred_ratings: user-defined and inferred rating for all meta-paths
        :return: Total order of all meta-paths with values in [0,1]
        """
        raise NotImplementedError()
        return []

    def fit_vectorizer(self, metapath_graph: RankingGraph) -> None:
        """

        :param metapath_graph: metapaths based on which to train the vectorizer.
        :return: Nothing.
        """
        self.vectorizer.fit([str(node) for node in metapath_graph.all_nodes()])

    def extract_features_labels(self, metapath_graph: RankingGraph) -> (List[Tuple], List):
        """
        Computes all pairwise tuples (a, b) of the meta-paths with their feature vector. If a is ranked higher than b
        a > b then the label is 1, 0 otherwise.

        :param metapath_graph: The meta-path graph representing the ordering of all meta-path
        :return: (x_train, y_train) The training feature vector and class labels.
        """

        metapath_pairs = []
        metapath_labels = []

        for closure in metapath_graph.transitive_closures():
            node_vector = closure.pop(0)
            for successor in closure:
                successor_vector = successor

                metapath_pairs.append((node_vector, successor_vector))
                metapath_labels.append(0)

                metapath_pairs.append((successor_vector, node_vector))
                metapath_labels.append(1)


        return metapath_pairs, metapath_labels
