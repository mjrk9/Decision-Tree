from scripts.entropy import *

class Node:
    def __init__(self, node_dict, full_matrix, bin_rep = '0', prune = False):
        """ Initialises an instance of a Node.

        Args:
            node_dict(dictionary): dictionary whose key is bin_rep and value is a Node
            full_mat(np array): dataset containing both the attribute matrix and the output labels
            bin_rep(str): string of binary representation (0 or 1) representing the location of a node.
                          '0' refers to a left branch and '1' refers to a right branch.

        For instance, a node with bin_rep '00' refers to the left branch of the left branch of the tree.
        The node with bin_rep '' refers to the root of the tree.
        """
        self.full_mat = full_matrix
        self.bin_rep = bin_rep
        self.node_dict = node_dict
        self.prune = prune

        self.split_val = 'N/A'
        self.att_col = 'N/A'

        self.left = None
        self.right = None
        self.left_node = None
        self.right_node = None
        self.isPruned= None

        classes, y = np.unique(full_matrix[:,-1], return_inverse=True)

        self.entropy = entropy(full_matrix[:,-1], classes)

        if not self.isTerminal():
            self.left, self.right, self.att_col, self.split_val = self.split_to_children()

   # def __repr__(self):
    #    return "Test a:% s b:% s" % (self.left, self.right)

    def fit(self):
        """ # Rephrase
        Recursively splits the dataset into left and right Nodes.
        """
        # If node is terminal, return itself and stop splitting.
        if self.isTerminal() or self.prune:
            self.node_dict[self.bin_rep] = self

        else:
            self.left_node = Node(self.node_dict, self.left, bin_rep = self.bin_rep+'0')
            self.left_node.fit()

            self.right_node = Node(self.node_dict, self.right, bin_rep = self.bin_rep+'1')
            self.right_node.fit()

            self.node_dict[self.bin_rep] = self


    def split_to_children(self):
        """
        Takes the current node's full matrix and splits it according to the optimal column and split value

        Returns:
            self.left (np array): array whose value in attribute column < split value
            self.right (np array): array whose value in attribute column >= split value
            self.att_col (int): index of attribute column which results in the most optimal split
            self.split_val (float): split value of attribute column which results in the most optimal split
        """
        self.att_col, self.split_val = self.find_split()
        _, left, right = calc_info_gain_of_matrix(self.full_mat, self.att_col, self.split_val)

        self.left = left
        self.right = right
        return self.left, self.right, self.att_col, self.split_val


    def isTerminal(self):
        """
        Checks if the current node is terminal i.e. if the leaves are pure.

        Returns:
            bool
        """
        if self.isPruned==True:
            return True
        else:
            y_class, _ = np.unique(self.full_mat[:,-1], return_inverse=True)
            return (abs(entropy(self.full_mat[:-1], y_class)) == 0)


    def find_split(self):
        """
        Chooses the best column and best split value for splitting.

        Returns:
            best_col (int): index of column to split by which results in highest information gain
            best_split_val (float): split value which of best_col to split by
        """
        best_gain = 0
        best_col = 0
        best_split_val = 0

        for col in range(self.full_mat.shape[1]-1):
            info_gain, split_val = find_gain_and_splitval(self.full_mat, col)

            if info_gain > best_gain:

                best_gain = info_gain
                best_col = col
                best_split_val = split_val

        return best_col, best_split_val

