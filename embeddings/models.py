import tensorflow as tf


def model_word2vec(features, labels, mode, params):
    """
    Word2vec model from "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al.)
    TODO: Add documentation
    :return:
    """
    assert features is not None, 'features is {}'.format(features)
    assert labels is not None, 'labels is {}'.format(labels)
    input = tf.feature_column.input_layer(features, params['feature_columns'])
    size_of_vocabulary = input.shape[1].value

    word_embeddings = tf.Variable(
        initial_value=tf.random_uniform(shape=[size_of_vocabulary, params['embedding_size']], minval=-1, maxval=1),
        name='word_embeddings')

    # Look up embedding for all words
    embedded_words = tf.nn.embedding_lookup(word_embeddings, tf.argmax(input, axis=1))
    assert embedded_words.shape == (
        input.shape[0].value, params['embedding_size']), 'Shape expected ({}, {}), but was {}'.format(
        input.shape[0].value, params['embedding_size'], embedded_words.shape)

    # Concatenate vectors
    concatenated_embeddings = tf.reshape(tf.concat(tf.unstack(embedded_words, axis=0), axis=0), shape=[1, -1])
    assert concatenated_embeddings.shape == (
        1, embedded_words.shape[0].value * embedded_words.shape[1].value), 'Shape expected ({}), but was {}'.format(
        1, embedded_words.shape[0].value * embedded_words.shape[1].value, concatenated_embeddings.shape)

    return _model_word2vec(mode, size_of_vocabulary, params['loss'], labels, concatenated_embeddings,
                           params['optimizer'])


def _model_word2vec(mode, size_of_vocabulary, loss: str, labels, concatenated_embeddings, optimizer: str):
    # Transform embeddings linearly
    hidden_layer = tf.layers.dense(inputs=concatenated_embeddings, units=size_of_vocabulary, activation=None,
                                   use_bias=True,
                                   name="linear_transformation",
                                   kernel_initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
    assert hidden_layer.shape == (
        1, size_of_vocabulary), 'Shape expected ({}, {}), but was {}'.format(
        1, size_of_vocabulary, hidden_layer.shape)

    # Apply softmax and calulate loss
    if loss == 'cross_entropy':
        # TODO: Reduce to one vector when using skip-grams (so shape is (1, 14) and not (1, 56))
        labels = tf.reshape(tf.one_hot(indices=labels, depth=size_of_vocabulary), shape=[1, -1])
        assert labels.shape == (
            1, size_of_vocabulary), 'Shape expected ({}, {}), but was {}'.format(
            1, size_of_vocabulary, labels.shape)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_layer, labels=labels),
                              name="cross_entropy_loss")
    else:
        raise NotImplementedError("This loss isn't implemented yet. Implement it!")

    total_train_op = __build_train_op(loss, optimizer)

    # Loss for tensorboard
    tf.summary.scalar('Loss', loss)

    metrics = {'loss': loss}
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=total_train_op)
    else:
        raise NotImplementedError("This mode isn't implemented yet")


def __build_train_op(loss, optimizer: str):
    if optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer()
    elif optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
    else:
        raise NotImplementedError("This optimizer isn't implemented yet. Implement it!")
    total_train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return total_train_op


def model_paragraph_vectors_skipgram(features, labels, mode, params):
    """
    Distributed Memory Model of Paragraph Vectors (PV-DM) from "Distributed Representations of Sentences and Documents" (Mikolov et al.)
    :return:
    """
    assert features is not None, 'features is {}'.format(features)
    assert labels is not None, 'labels is {}'.format(labels)
    context = tf.feature_column.input_layer(features, params['feature_columns'])
    print(params['paragraph_columns'])
    print(features.values)
    paragraph = tf.feature_column.input_layer(features, params['paragraph_columns'])
    size_of_node_vocabulary = context.shape[1].value
    size_of_paragraph_vocabulary = paragraph.shape[1].value

    node_embeddings = tf.Variable(
        initial_value=tf.random_uniform(shape=[size_of_node_vocabulary, params['embedding_size'][0]], minval=-1,
                                        maxval=1),
        name='word_embeddings')
    paragraph_embedding = tf.Variable(
        initial_value=tf.random_uniform(shape=[size_of_paragraph_vocabulary, params['embedding_size'][1]], minval=-1,
                                        maxval=1),
        name='paragraph_embeddings')

    # Look up embedding for all words
    embedded_words = tf.nn.embedding_lookup(node_embeddings, tf.argmax(context, axis=1))
    embedded_paragraph = tf.nn.embedding_lookup(paragraph_embedding, tf.argmax(paragraph, axis=1))
    assert embedded_words.shape == (
        context.shape[0].value, params['embedding_size'][0]), 'Shape expected ({}, {}), but was {}'.format(
        context.shape[0].value, params['embedding_size'][0], embedded_words.shape)
    assert embedded_paragraph.shape == (
        paragraph.shape[0].value, params['embedding_size'][1]), 'Shape expected ({}, {}), but was {}'.format(
        paragraph.shape[0].value, params['embedding_size'][1], embedded_paragraph.shape)

    # Concatenate vectors
    concatenated_embeddings = tf.reshape(
        tf.concat(
            [embedded_paragraph,
             [tf.concat(
                 tf.unstack(embedded_words, axis=0),
                 axis=0,
                 name="concat_words")]],
            axis=1,
            name="concat_all"),
        shape=[1, -1])
    assert concatenated_embeddings.shape == (
        1, embedded_words.shape[0].value * embedded_words.shape[1].value + embedded_paragraph.shape[
            1].value), 'Shape expected ({}), but was {}'.format(
        (1, embedded_words.shape[0].value * embedded_words.shape[1].value + embedded_paragraph.shape[1].value),
        concatenated_embeddings.shape)

    return _model_word2vec(mode, size_of_node_vocabulary, params['loss'], labels, concatenated_embeddings,
                           params['optimizer'])


def model_paragraph_vectors_dbow(features, labels, mode, params):
    """
    Distributed Bag of Words version of Paragraph Vector (PV-DBOW) from "Distributed Representations of Sentences and Documents" (Mikolov et al.)
    :return:
    """
    raise NotImplementedError("This model isn't implemented yet")
