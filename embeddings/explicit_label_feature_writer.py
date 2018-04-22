from embeddings.randomwalk_embedder import *
import numpy as np

# File parameters
walks_file_name = "mock_input.txt"


# Generator parameters
skip_window = 3        # size of the window to the left/right
num_skips = 5          # how many times to reuse a walk to generate a label

# Working parameters (affecting output in length)
batch_size = 60
iterations = 1




walk_list, available_nodes = simple_read_walks(file_name=walks_file_name)
batch_generator = LongWalkBatchGenerator(walk_list, skip_window=skip_window)


def write_context_to_file(np_array, file):
    np.apply_along_axis(lambda array: (file.write(np_array_to_string(array))), 1, np_array)


def np_array_to_string(np_array):
    string = ''
    for number in np.nditer(np_array):
        string = string  + ' ' + str(number)
    return string + '\n'


def write_labels_to_file(np_array, file):
    for number in np.nditer(np_array):
        file.write(str(number) + '\n')


with open('labels.txt', 'w') as labels_file, open('features.txt', 'w') as features_file:

    for _ in range(iterations):
        batch_inputs, batch_context = batch_generator.generate_batch(batch_size, num_skips)

        print(batch_context)
        # np.apply_along_axis(lambda array: (print(array)), 1, batch_context)

        write_labels_to_file(batch_inputs, labels_file)
        write_context_to_file(batch_context, features_file)
