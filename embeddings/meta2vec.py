from typing import Tuple
import tensorflow as tf


class Input:
    @classmethod
    def from_json(cls, data):
        raise NotImplementedError()

    def input(self) -> Tuple:
        raise NotImplementedError()


class MetaPathsInput(Input):

    def __init__(self, dataset):
        self.dataset = self.apply_transformation(dataset)

    @classmethod
    def from_json(cls, data):
        dataset = tf.data.Dataset()

        return cls(dataset)

    def apply_transformation(self, dataset):
        pass

    @classmethod
    def parse_meta_paths(json, min_size=5, seperator=" | "):
        walk_list = []
        available_nodes = set()
        for meta_paths in json.keys():
            node_ids = [int(id) for id in meta_paths.split(seperator)]
            if (len(node_ids) < min_size):
                continue
            walk_list.append(node_ids)
            available_nodes |= set(node_ids)
        return walk_list, available_nodes

    def input(self):
        return self.dataset


class NodeInput(Input):
    pass


def model_word2vec(features, labels, mode, params):
    """
    Word2vec model from "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al.)
    :return:
    """
    input = tf.feature_column.input_layer(features, params['feature_columns'])

    size_of_vocabulary = input.shape[1]

    word_embeddings = tf.Variable(
        initial_value=tf.random_uniform(shape=[size_of_vocabulary, params['embedding_size']], minval=-1, maxval=1),
        name='word_embeddings')

    # Look up embedding for all words
    embedded_words = tf.nn.embedding_lookup(word_embeddings, tf.argmax(input, axis=0))

    return _model_word2vec(mode, size_of_vocabulary, params['loss'], labels, embedded_words)


def _model_word2vec(mode, size_of_vocabulary, loss, labels, embedded_words):
    # Concatenate vectors
    concatenated_embeddings = embedded_words

    # Transform embeddings linearly
    hidden_layer = tf.layers.dense(inputs=concatenated_embeddings, units=size_of_vocabulary, activation=None,
                                   use_bias=True,
                                   name="linear_transformation",
                                   kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

    # Apply softmax and calulate loss
    if loss == 'cross_entropy':
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_layer, labels=labels),
                              name="cross_entropy_loss")
    else:
        raise NotImplementedError("This loss isn't implemented yet. Implement it!")

    optimizer = tf.train.AdamOptimizer()
    total_train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    metrics = {'loss': loss}
    # Loss for tensorboard
    tf.summary.scalar('Loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=total_train_op)
    else:
        raise NotImplementedError("This mode isn't implemented yet")


def model_paragraph_vectors_skipgram():
    """
    Distributed Memory Model of Paragraph Vectors (PV-DM) from "Distributed Representations of Sentences and Documents" (Mikolov et al.)
    :return:
    """
    pass


def model_paragraph_vectors_dbow():
    """
    Distributed Bag of Words version of Paragraph Vector (PV-DBOW) from "Distributed Representations of Sentences and Documents" (Mikolov et al.)
    :return:
    """
    pass


def create_estimator():
    pass


def parse_arguments():
    pass


if __name__ == "__main__":
    args = parse_arguments()
    classifier = create_estimator()

    if args.mode == 'train':
        # TODO: Create Dataset
        dataset = tf.data.Dataset()
        classifier.train(input_fn=MetaPathsInput(dataset).input())
    elif args.mode == 'predict':
        raise NotImplementedError()
    elif args.mode == 'eval':
        raise NotImplementedError()
