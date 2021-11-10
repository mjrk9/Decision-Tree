import numpy as np
import matplotlib.pyplot as plt


def calc_info_gain_of_matrix(full_mat, att_col, split_val, show = False):
    """
    Takes the full matrix, chosen attribute column (by index) and value by which to split.
    Return children decides if it will return the Left and Right matrices as well (which would be
    full matrices of the children nodes)
    """

    left, right = split_matrix(full_mat, att_col, split_val)
    left_spec_col = left[:, [att_col, -1]]
    right_spec_col = right[:, [att_col, -1]]
    y_class, y_labels = np.unique(full_mat[:,-1], return_inverse=True)
    info_gain = information_gain(full_mat[:,-1], left_spec_col[:,1], right_spec_col[:,1], y_class)

    return info_gain, left, right


def split_matrix(full_mat, att_col, split_val):
    """
    Splits the matrix by the attribute column and along the split value
    """
    left = full_mat[(split_val > full_mat[:,att_col])]
    right = full_mat[(split_val <= full_mat[:,att_col])]
    return left, right


def append_y_to_atts(attribute_matrix, y):
    """
    Adds the group vector (y) to the furthest right
    column of the attribute matrix x
    """
    y = y.reshape((len(y),1))
    att_mat = np.array(attribute_matrix)
    att_mat = np.append(att_mat, y, axis=1)
    return att_mat


def find_gain_and_splitval(full_mat, att_col):
    """
    Finds the best value to split the given matrix by (for chosen attribute column)
    """
    attribute_values = full_mat[:,att_col]
    sorted_vals = np.unique(attribute_values)

    max_gain = -1000
    max_val = 0

    for i in range(len(sorted_vals)):
        info_gain, left, right = calc_info_gain_of_matrix(full_mat, att_col, sorted_vals[i])
        if info_gain > max_gain:
            max_gain = info_gain
            max_val = sorted_vals[i]

    return max_gain, max_val


def FIND_SPLIT(full_mat):
    ''' Chooses the split point (attribute and value) that results in the highest information gain.

    Args:
        data(matrix): dataset

    Returns:
        attribute(int): index of attribute in the dataset to split by that results in highest information gain
        value(float): value of attribute in the dataset to split by that results in highest information gain

    Adapted from the original function; takes the full matrix as an input and finds the optimal split
    rather than adding the y column in this function itself. Makes it easier when splitting so we dont
    have to worry about the indices of the y column to split as well.
    '''
    split_nested_list = []
    for col in range(full_mat.shape[1]-1):
        info_gain, split_val = find_gain_and_splitval(full_mat, col)
        split_nested_list.append([col, info_gain, split_val])

    split_nested_list = np.array(split_nested_list)
    best_split = np.argmax(split_nested_list[:,1])
    best_split_value = split_nested_list[best_split, 2]

    return best_split, best_split_value


def decision_tree_learning(data, depth):
    '''
    Args:
        data(matrix): dataset
        depth(int): maximal depth of tree

    Returns:

    '''
    _, best_split, best_split_value, best_info_gain = FIND_SPLIT(data)
    pass



def plot_gain(attribute_matrix, att_col, y):
    """
    Little plot function if we want to test our search algorithm for optimal split value
    """

    attribute_values = attribute_matrix[:,att_col]
    sorted_vals = np.unique(attribute_values)

    left = 0
    right = len(sorted_vals)-1
    split_count = 0

    gains = list(map(lambda x: calc_info_gain(attribute_matrix, att_col, x, y), sorted_vals))

    plt.plot(sorted_vals, gains)
    plt.xlabel('Split value')
    plt.ylabel('Information gain')
    plt.savefig('info gain.jpg')


def entropy(y, classes):
    ''' Calculates entropy for a multi-class dataset.

    Args:
        y (1-d array/list): output labels
        classes:  unique class labels corresponding to the integers in y

    Returns:
        entropy(float): entropy

    '''
    probability_class = []
    entropy_class = []

    if len(y) == 0:
        return 0

    for feature in classes:
        probability = len(y[y == feature])/ len(y)
        if probability == 1 or probability == 0:
            probability_class.append(probability)
            entropy_class.append(0)
            continue
        probability_class.append(probability)
        entropy_class.append(np.log2(probability))
    entropy = np.dot(np.negative(probability_class), entropy_class)
    return entropy


def information_gain(data, l_branch, r_branch, y_classes):
    ''' Calculates information gain for a particular split.

    Args:
        data(1-d array): y column of original dataset
        l_branch(1-d array): y column of left branch of the original data
        r_branch(1-d array): y column of right branch of the original data
        y classes


    Returns:
        gain (float): information gain
    '''

    entropy_main = entropy(data, y_classes)
    entropy_left = entropy(l_branch, y_classes)
    entropy_right = entropy(r_branch, y_classes)

    len_left = len(l_branch)
    len_right = len(r_branch)

    remainder = len_left/(len_left + len_right) * entropy_left + len_right/(len_left + len_right) * entropy_right
    gain = entropy_main - remainder

    return gain


def read_dataset(filepath):
    """ Read in the dataset from the specified filepath

    Args:
        filepath (str): The filepath to the dataset file

    Returns:
        tuple: returns a tuple of (x, y, classes), each being a numpy array.
               - x is a numpy array with shape (N, K),
                   where N is the number of instances
                   K is the number of features/attributes
               - y is a numpy array with shape (N, ), and should be integers from 0 to C-1
                   where C is the number of classes
               - classes : a numpy array with shape (C, ), which contains the
                   unique class labels corresponding to the integers in y
    """
    x = []
    y_labels = []
    for line in open(filepath):
        if line.strip() != "": # handle empty rows in file
            row = line.split('\t')
            x.append(list(map(float, row[:-1])))
            y_labels.append(row[-1])

    classes, y = np.unique(y_labels, return_inverse=True)
    new_classes = np.array([])
    for k in classes:
        new_classes = np.append(new_classes, int(k))

    classes = new_classes

    x = np.array(x)
    y = np.array(y + 1)
    return (x, y, classes)
