from typing import List, Tuple
import numpy
from util.ranking_graph import RankingGraph
from util.datastructures import MetaPath
from util.lists import all_pairs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from domain_scoring.domain_value_transformer import NaiveTransformer, SMALLER, LARGER
import util.config as config

class DomainScoring():
    def __init__(self):
        """
        Classifies the ordering and extracts the domain value of meta-paths.
        """
        # The token_pattern also allows single character strings which the default doesn't allow
        self.vectorizer = TfidfVectorizer(analyzer='word', token_pattern='\\b\\w+\\b')
        self.classifier = DecisionTreeClassifier()
        self.domain_value_transformer = NaiveTransformer()


    def fit(self, metapath_graph: RankingGraph) -> None:
        """
        Fits a classifier to predict a meta-path ordering.
        :param metapath_graph: already ordered meta-path used as a training set.
        :return: Nothing.
        """
        self._fit_vectorizer(metapath_graph)
        x_train, y_train = self._extract_training_data_labels(metapath_graph)

        self.classifier.fit(self._preprocess(x_train), y_train)



    def predict(self, metapath_unrated: List[MetaPath]) -> List[Tuple[MetaPath, int]]:
        """
        Predict the domain value of the given meta-paths.
        :param metapath_unrated: The meta-paths to be ordered.
        :return: A list of tuples with the metapath and their predicted domain value.
        """
        x_predict = all_pairs(metapath_unrated)
        y_predict = self.classifier.predict(self._preprocess(x_predict))


        return self.transform_to_domain_values(x_predict, y_predict)

    def _preprocess(self, data: List[Tuple[MetaPath, MetaPath]]) -> List[List[int]]:
        """
        Takes a list of metapaths pairs (a, b) and vectorizes a and b and joins them to one element for training.
        :param data: the data to preprocess.
        :return: preprocessed data.
        """
        vectors = []
        for datum in data:
            feature_matrix = numpy.concatenate(self.vectorizer.transform(map(str, datum)).toarray())
            vectors.append(feature_matrix.tolist())

        return vectors

    def _transform_to_domain_values(self, metapaths_pairs: List[Tuple[MetaPath, MetaPath]], classification: List[int]) -> List:
        """
        Transforms the classified ordering of all meta-paths pairs to the domain values.

        :param inferred_ratings: user-defined and inferred rating for all meta-paths
        :return: Total order of all meta-paths with values in [0,10]
        """

        return self.domain_value_transformer.transform(metapaths_pairs, classification)

    def _fit_vectorizer(self, metapath_graph: RankingGraph) -> None:
        """

        :param metapath_graph: metapaths based on which to train the vectorizer.
        :return: Nothing.
        """
        self.vectorizer.fit([str(node) for node in metapath_graph.all_nodes()])

    def _extract_training_data_labels(self, metapath_graph: RankingGraph) -> (List[Tuple[MetaPath]], List[int]):
        """
        Computes all pairwise tuples (a, b) of the meta-paths with their feature vector. If a is ranked higher than b
        a > b then the label is 1, 0 otherwise.

        :param metapath_graph: The meta-path graph representing the ordering of all meta-path
        :return: (x_train, y_train) The training feature vector and class labels.
        """

        metapath_pairs = []
        metapath_labels = []

        for closure in metapath_graph.transitive_closures():
            node_vector = closure.pop(0) # pop first element
            for successor in closure:
                successor_vector = successor

                metapath_pairs.append((node_vector, successor_vector))
                metapath_labels.append(SMALLER) # <

                metapath_pairs.append((successor_vector, node_vector))
                metapath_labels.append(LARGER) # >


        return metapath_pairs, metapath_labels
