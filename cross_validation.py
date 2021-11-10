import numpy as np
from scripts.entropy import *

# Attempting 10-fold cross validation

seed = 65650

rng = np.random.default_rng(seed) # Does not work with older versions of numpy


# First split 2000 indices into 10 parts
def n_fold_split(n_splits, n_instances, random_gen=rng):
    """ Split n_instances into n mutually exclusive splits at random.

    Args:
        n_splits (int): Number of splits (10 in our case)
        n_instances (int): Number of instances to split (2000 in our case)
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list (length n_splits). Each element in the list should contain a
            numpy array giving the indices of the instances in that split.
    """

    # Generate randomised order of the n_instances indices
    shuffled_indices = random_gen.permutation(n_instances)

    # Split the indices into n_split (equal) parts
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices


# Now implement cross validation
def train_test_n_fold(n_folds, n_instances, random_gen=rng):
    """ Generate train and test indices at each fold.

    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple)
            with two elements: a numpy array containing the train indices, and another
            numpy array containing the test indices.
    """

    # split the dataset into k splits
    split_indices = n_fold_split(n_folds, n_instances, random_gen)

    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]

        # combine remaining splits as train
        train_indices = np.hstack(split_indices[:k] + split_indices[k + 1:])

        folds.append([train_indices, test_indices])

    return folds


def generate_CV_data(x, y): 
    full_matrix = append_y_to_atts(x, y)
    folded_data = []
    for (train_indices, test_indices) in train_test_n_fold(10, len(y)): 
        # print(train_indices)
        train_data = full_matrix[train_indices, :]
        test_data = full_matrix[test_indices, :]
        folded_data.append([train_data, test_data])

    return np.array(folded_data)

    



# # Testing 5 fold with 50 instances
# for (train_indices, test_indices) in train_test_n_fold(5, 50, rng):
#     print("train: ", train_indices)
#     print("test: ", test_indices)
#     print()
