import tensorflow as tf


def create_word2vec_estimator(vocab_size: int, model_fn, model_dir, embedding_size: int, optimizer: str, loss: str,
                              gpu_memory: float):
    features = tf.feature_column.categorical_column_with_hash_bucket('features',
                                                                     vocab_size,
                                                                     dtype=tf.int32)
    indicator_column = tf.feature_column.indicator_column(features)
    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
    run_config = tf.estimator.RunConfig(tf_random_seed=42,
                                        save_summary_steps=500,
                                        save_checkpoints_steps=1000,
                                        keep_checkpoint_max=5,
                                        keep_checkpoint_every_n_hours=0.25,
                                        log_step_count_steps=50,
                                        session_config=session_config)
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params={'feature_columns': [indicator_column],
                'embedding_size': embedding_size,
                'optimizer': optimizer,
                'loss': loss},
        config=run_config)
    return classifier


def create_paragraph_estimator(model_dir, model_fn, node_count: int, paths_count: int, embedding_size: int,
                               optimizer: str, loss: str,
                               gpu_memory: float):
    print(node_count)
    context = tf.feature_column.categorical_column_with_hash_bucket('features',
                                                                    node_count,
                                                                    dtype=tf.int32)

    print(paths_count)
    paragraph = tf.feature_column.categorical_column_with_hash_bucket('paragraphs',
                                                                      paths_count,
                                                                      dtype=tf.int32)

    context_indicator = tf.feature_column.indicator_column(context)
    paragraph_indicator = tf.feature_column.indicator_column(paragraph)
    session_config = tf.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory
    run_config = tf.estimator.RunConfig(tf_random_seed=42,
                                        save_summary_steps=500,
                                        save_checkpoints_steps=1000,
                                        keep_checkpoint_max=5,
                                        keep_checkpoint_every_n_hours=0.25,
                                        log_step_count_steps=50,
                                        session_config=session_config)
    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        params={'feature_columns': [context_indicator],
                'paragraph_columns': [paragraph_indicator],
                'embedding_size': embedding_size,
                'optimizer': optimizer,
                'loss': loss},
        config=run_config)
    return classifier
