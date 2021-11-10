import numpy as np

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
        if line.strip() != "":  # handle empty rows in file
            row = line.split('\t')
            x.append(list(map(float, row[:-1])))
            y_labels.append(row[-1])

    _, y = np.unique(y_labels, return_inverse= True)
    y = np.array(y + 1)
    x = np.array(x)
    return (x, y)

class DecisionTree:
    def __init__(self):
        pass

    def fit(self, x, y):
        """ Fits training data to the classifier.
        """
        self.x = x
        self.y = y
        self.classes = set(self.y)

    def predict(self, x):
        """ Performs predictions given some examples
        """
        pass

    def entropy(self, y, classes):
        ''' Calculates entropy for a multi-class dataset.

        Args:
            y (1-d array/list): output labels
            classes:  unique class labels corresponding to the integers in y

        Returns:
            entropy(float): entropy

        '''
        probability_class = [] # Probability of being in a class k
        entropy_class = []  # Entropy of being in a class k

        if len(y) == 0:
            return 0

        for k in classes:
            probability = len(y[y == k]) / len(y)
            if probability == 1 or probability == 0:
                probability_class.append(probability)
                entropy_class.append(0)
                continue

            probability_class.append(probability)
            entropy_class.append(np.log2(probability))

        entropy = np.dot(np.negative(probability_class), entropy_class)
        return entropy

    def calc_info_gain(self, y, l_branch, r_branch, classes):
        ''' Calculates information gain for a particular split.

        Args:
            y(1-d array): output y
            l_branch(1-d array): output y of left branch
            r_branch(1-d array): output y of right branch
            classes: unique class labels
        Returns:
            gain (float): information gain
        '''
        entropy_main = self.entropy(y, classes)
        entropy_left = self.entropy(l_branch, classes)
        entropy_right = self.entropy(r_branch, classes)

        # print("entropy_main: ", entropy_main)
        # print("entropy_left: ", entropy_left)
        # print("entropy_right: ", entropy_right)

        len_left = len(l_branch)
        len_right = len(r_branch)

        remainder = len_left / (len_left + len_right) * entropy_left + len_right / (
                    len_left + len_right) * entropy_right
        gain = entropy_main - remainder
        # print("info gain: ", gain)
        return gain

    def calc_info_gain_for_col(self, x, x_col_index, split_val, y):
        """
        Takes attribute matrix, column of the attribute we want to split by,
        the value of that attribute we split at, and the features of the inputs.

        For a certain attribute it will split the matrix into those with values
        >= and < split_val and put these into two separate children matrices.
        The entropy of these can then be calculated and used to calculate the
        information gain from the initial y column.

        Args:
            x: numpy array (size N by K) to represent features of data
            x_col_index(int): index of column to split on. Starts from 0
            split_val (float): value of attribute to split on
            y (1-d array): output labels

        Returns:
            info_gain(float): information gain
        """

        classes = set(y)
        y = y.reshape((len(y), 1))
        x_col = np.array([x[:, x_col_index]])
        x_col = np.transpose(x_col)

        xy_col = np.append(x_col, y, axis=1)  # matrix containing column of attribute and y values

        # print("split_val:", split_val)
        left = xy_col[(split_val > xy_col[:, 0])] # strictly less than split value
        right = xy_col[(split_val <= xy_col[:, 0])]
        # print("left: ", left)
        # print("right: ", right)

        info_gain = self.calc_info_gain(y, left[:, 1], right[:, 1], classes)

        return info_gain

    def find_opt_split_val_for_col(self, x, x_col_index, y):
        """ Find optimal split value for a column.

        May change this for a more efficient implementation!

        Uses an exhaustive search. Loops through all possible values of a column of x,
        and calculates the corresponding information gain for each split value.

        Returns the optimal split value and the optimal information gain.

        Args:
            x: numpy array (size N by K) to represent features of data
            x_col_index(int): index of column to split on. Starts from 0
            y (1-d array): output labels

        Returns:
            opt_split_val (int/float): optimal split value that yields the highest information gain
            opt_info_gain(float): optimal (highest) information gain
        """
        x_col = np.array([x[:, x_col_index]])
        x_col = np.transpose(x_col)
        # print(x_col_index, x_col)
        x_col = np.reshape(x_col, len(x_col)) # converted to 1-d array, may want to change back later
        x_values = list(set(x_col))
        x_values.sort()

        info_gain_list = []
        for i in list(x_values)[1:]: # Start from 2nd smallest element
            if self.calc_info_gain_for_col(x, x_col_index, i, y) == None:
                info_gain_list.append(-10000)
            else:
                info_gain_list.append(self.calc_info_gain_for_col(x, x_col_index, i, y))
            # print("--------------")
        # print("I", info_gain_list)
        opt_info_gain = np.max(info_gain_list)
        opt_idx = np.argmax(info_gain_list)
        opt_split_val = x_values[opt_idx+1]

        return opt_split_val, opt_info_gain

    def find_split(self, x, y):
        ''' Chooses the split point (attribute and value) that results in the highest information gain.

        Args:
            x:
            y:

        Returns:
            best_split_index(int): index of attribute in x to split by that results in highest information gain
            best_split_value(float): value of attribute in x to split by that results in highest information gain
            best_info_gain(float): optimal (highest) value of information gain
        '''

        split_nested_list = []
        for x_col_index in range(x.shape[1]):
            if (np.max(x[:, x_col_index]) == np.min(x[:, x_col_index])):
                split_nested_list.append([x_col_index, None, float('-inf')])
            else:
                split_val, info_gain = self.find_opt_split_val_for_col(x, x_col_index, y)
                split_nested_list.append([x_col_index, split_val, info_gain])

        split_nested_list = np.array(split_nested_list)

        # Best column to split on
        best_split_index = np.argmax(split_nested_list[:,2])
        best_info_gain = np.max(split_nested_list[:,2])
        best_split_value = split_nested_list[best_split_index,1]

        # Store information as node - keep track of left and right branches
        # Inefficient because we have done this before
        # But I don't want to store information when calculating optimal values

        y = y.reshape((len(y), 1))
        xy = np.append(x, y, axis=1)

        x_col = np.array(xy[:, best_split_index])
        x_col = np.transpose(x_col)
        left = xy[x_col < best_split_value]
        right = xy[x_col >= best_split_value]

        node = {'attribute': best_split_index,
                'value': best_split_value,
                'info gain': best_info_gain,
                'left': left,
                'right': right}

        return node
    def to_terminal(self, group):
        y = group[:, -1]
        return self.find_mode(y)

    def find_mode(self,array):
        vals, counts = np.unique(array, return_counts= True)
        idx = np.argmax(counts)
        return vals[idx]

    def decision_tree_learning(self, node, max_depth, min_size, depth):
        left = node["left"]
        right = node["right"]
        del(node["left"])
        del(node["right"])

        if left.size ==0 or right.size == 0:
            print("a")
            node["left"] = node["right"] = self.to_terminal(left + right) # may have to append properly
            return

        if depth >= max_depth:
            print("b")
            node["left"], node["right"] = self.to_terminal(left), self.to_terminal(right)
            return

        if len(left) <= min_size:
            print("c")
            node["left"] = self.to_terminal(left)

        else:
            print("d")
            # print("left", left)
            node["left"] = self.find_split(left[:,0:-1], left[:,-1])
            self.decision_tree_learning(node['left'], max_depth, min_size, depth+1)

        if len(right) <= min_size:
            print("e")
            node["right"] = self.to_terminal(right)

        else:
            print("f")
            node["right"] = self.find_split(right[:,0:-1], right[:,-1])
            self.decision_tree_learning(node['right'], max_depth, min_size, depth+1)


    def build_tree(self, max_depth, min_size = 3):
        root = self.find_split(self.x, self.y) # Returns me a node
        print("root", root)
        self.decision_tree_learning(root, max_depth, min_size, 1)
        return root






#####################################
#####################################
#####################################

## Run printing code below this line
x, y = read_dataset("wifi_db/clean_dataset.txt")
# print(x, y)

decision_tree = DecisionTree()
decision_tree.fit(x, y)
tree = decision_tree.build_tree(10)

print(tree)

# Entropy function test
# print(decision_tree.entropy(y, set(y)))

### Test code for small data
# x_test = np.array([[0,1,2,3,4,5],
#                   [1,7,8,9,10,11],
#                   [2,1,2,3,4,5],
#                   [3,1,2,3,4,5],
#                   [4,1,8,3,4,5]])
#
# y_test = np.array([0,1,0,0,1])
# print(x_test.shape, y_test.shape) # note that y is of shape (5,) which is the same as our datasset y
# decision_tree.fit(x_test, y_test)
# print(decision_tree.calc_info_gain_for_col(x_test, 2, split_val = 8, y =y_test))

# print(decision_tree.find_opt_split_val_for_col(x_test, 0, y_test))
# print(decision_tree.find_split(x_test, y_test))

# print(decision_tree.build_tree(5))

