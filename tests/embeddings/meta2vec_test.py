import unittest
import tensorflow as tf
from embeddings.meta2vec import choose_function
from embeddings.estimators import create_word2vec_estimator, create_paragraph_estimator


class CreateEstimatorTest(unittest.TestCase):
    model_dir = 'test-model'
    #json_path_edges = '../data/mock_metapaths_edges.json'
    json_path = '../data/mock_metapaths.txt'

    def test_estimator_word2vec_bag_of_words(self):
        embedding_size = 5
        loss = 'adam'
        gpu_memory = 0.3
        optimizer = 'adam'
        _, model_fn, _ = choose_function(model='word2vec', model_type="bag-of-words",
                                         input_type='',
                                         json_path=self.json_path)
        self.assertEqual(tf.estimator.Estimator,
                         create_word2vec_estimator(vocab_size=10, embedding_size=embedding_size, loss=loss,
                                                   gpu_memory=gpu_memory, model_fn=model_fn,
                                                   model_dir=self.model_dir, optimizer=optimizer).__class__)

    def test_estimator_word2vec_skip_gram(self):
        embedding_size = 5
        loss = 'adam'
        gpu_memory = 0.3
        optimizer = 'adam'
        _, model_fn, _ = choose_function(model='word2vec', model_type="skip-gram",
                                         input_type='',
                                         json_path=self.json_path)
        self.assertEqual(tf.estimator.Estimator,
                         create_word2vec_estimator(vocab_size=10, embedding_size=embedding_size, loss=loss,
                                                   gpu_memory=gpu_memory, model_fn=model_fn,
                                                   model_dir=self.model_dir, optimizer=optimizer).__class__)

    def test_estimator_paragraph_vectors_bag_of_words(self):
        embedding_size = 5
        loss = 'adam'
        gpu_memory = 0.3
        optimizer = 'adam'
        _, model_fn, _ = choose_function(model='paragraph_vectors', model_type="bag-of-words",
                                         input_type='',
                                         json_path=self.json_path)
        self.assertEqual(tf.estimator.Estimator,
                         create_paragraph_estimator(node_count=10, paths_count=20, embedding_size=embedding_size, loss=loss,
                                                   gpu_memory=gpu_memory, model_fn=model_fn,
                                                   model_dir=self.model_dir, optimizer=optimizer).__class__)

    def test_estimator_paragraph_vectors_skip_gram(self):
        embedding_size = 5
        loss = 'adam'
        gpu_memory = 0.3
        optimizer = 'adam'
        _, model_fn, _ = choose_function(model='paragraph_vectors', model_type="skip-gram",
                                         input_type='',
                                         json_path=self.json_path)
        self.assertEqual(tf.estimator.Estimator,
                         create_paragraph_estimator(node_count=10, paths_count=20, embedding_size=embedding_size, loss=loss,
                                                   gpu_memory=gpu_memory, model_fn=model_fn,
                                                   model_dir=self.model_dir, optimizer=optimizer).__class__)
