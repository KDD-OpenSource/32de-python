from typing import List, Tuple
import numpy
from sklearn.ensemble import RandomForestRegressor

from util.datastructures import MetaPathRatingGraph
from util.datastructures import MetaPath
from util.lists import all_pairs
from util.config import RANDOM_STATE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from domain_scoring.domain_value_transformer import NaiveTransformer, SMALLER, LARGER

Ranking = Tuple[MetaPath, float]


class DomainScoring():
    def __init__(self):
        """
        Classifies the ordering and extracts the domain value of meta-paths.
        """
        # The token_pattern also allows single character strings which the default doesn't allow
        self.vectorizer = TfidfVectorizer(analyzer='word', token_pattern='\\b\\w+\\b')
        self.random_state = RANDOM_STATE
        self.classifier = DecisionTreeClassifier(random_state=self.random_state)
        self.domain_value_transformer = NaiveTransformer()

    def fit(self, metapath_graph: MetaPathRatingGraph, test_size: float = None) -> None:
        """
        Fits a classifier to predict a meta-path ordering.
        :param metapath_graph: already ordered meta-path used as a training set.
        :param test_size: Specify size of test set if a test accuracy should be reported.
                          If empty or None is specified no accuracy is reported.
        :return: Nothing.
        """
        self._fit_vectorizer(metapath_graph)
        x, y = self._extract_data_labels(metapath_graph)

        if test_size is not None:
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                                test_size=test_size,
                                                                random_state=self.random_state,
                                                                shuffle=True)
        else:
            x_train = x
            y_train = y

        self.classifier = self.classifier.fit(self._preprocess(x_train), y_train)

        if test_size:
            self._test_score(x_test, y_test)

    def predict(self, metapath_unrated: List[MetaPath]) -> List[Tuple[MetaPath, int]]:
        """
        Predict the domain value of the given meta-paths.
        :param metapath_unrated: The meta-paths to be ordered.
        :return: A list of tuples with the metapath and their predicted domain value.
        """
        x_predict = all_pairs(metapath_unrated)
        y_predict = self.classifier.predict(self._preprocess(x_predict))

        return self._transform_to_domain_values(x_predict, y_predict)

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

    def _transform_to_domain_values(self,
                                    metapaths_pairs: List[Tuple[MetaPath, MetaPath]],
                                    classification: List[int]) -> List[Ranking]:
        """
        Transforms the classified ordering of all meta-paths pairs to the domain values.

        :param inferred_ratings: user-defined and inferred rating for all meta-paths.
        :return: Total order of all meta-paths with values in [0,1].
        """

        return self.domain_value_transformer.transform(metapaths_pairs, classification)

    def _fit_vectorizer(self, metapath_graph: MetaPathRatingGraph) -> None:
        """

        :param metapath_graph: metapaths based on which to train the vectorizer.
        :return: Nothing.
        """
        self.vectorizer.fit([str(node) for node in metapath_graph.all_nodes()])

    def _extract_data_labels(self, metapath_graph: MetaPathRatingGraph) -> (List[Tuple[MetaPath]], List[int]):
        """
        Computes all pairwise tuples (a, b) of the meta-paths. If a is ranked higher than b
        a > b then the label is 1, 0 otherwise.

        :param metapath_graph: The meta-path graph representing the ordering of all meta-path.
        :return: (x, y) The feature vector and class labels.
        """

        metapath_pairs = []
        metapath_labels = []

        for superior, inferior, distance in metapath_graph.stream_meta_path_distances():
            metapath_pairs.append((inferior, superior))
            metapath_labels.append(SMALLER)  # <

            metapath_pairs.append((superior, inferior))
            metapath_labels.append(LARGER)  # >

        return metapath_pairs, metapath_labels

    def _test_score(self, x_test, y_test):
        print('Test accuracy is {}'.format(self.classifier.score(X=self._preprocess(x_test), y=y_test)))

class DomainScoringRegressor(DomainScoring):

    def __init__(self):
        """
        Extracts the domain value of meta-paths via regression.
        """
        super().__init__()
        self.classifier = RandomForestRegressor(random_state=self.random_state)

    def _extract_data_labels(self, metapath_graph: MetaPathRatingGraph) -> (List[Tuple[MetaPath]], List[int]):
        """
        Computes all pairwise distances (a, b) of the meta-paths.

        :param metapath_graph: The meta-path graph representing the ordering of all meta-path.
        :return: (x, y) The meta-paths pairs and their respective distance.
        """

        metapath_pairs = []
        metapath_labels = []

        for superior, inferior, distance in metapath_graph.stream_meta_path_distances():
            metapath_pairs.append((inferior, superior))
            metapath_labels.append(distance)  # <

            metapath_pairs.append((superior, inferior))
            metapath_labels.append(-distance)  # >

        return metapath_pairs, metapath_labels

    def _test_score(self, x_test, y_test):
        """
        Converts regression result into a binary classification and uses mean accuracy.
        """
        test_predict = self.classifier.predict(self._preprocess(x_test))
        score = numpy.mean(numpy.logical_and(numpy.array(y_test) > 0, numpy.array(test_predict) > 0))
        print('Test accuracy is {}'.format(score))
        print('R^2 is {}'.format(self.classifier.score(X=self._preprocess(x_test), y=y_test)))