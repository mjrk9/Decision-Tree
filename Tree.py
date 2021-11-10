from scripts.entropy import *
from scripts.Node import Node
import matplotlib.pyplot as plt 

class Tree():
    def __init__(self, *args):
            self.node_dict = {}
            if args != None:
             for ar in args:
                self.node_dict=ar

    def fit(self, x, y):
        """
        Reads in the dataset and stores them as training data.

        Fits a decision tree to the training data.

        Args:
            x (np array): attribute matrices for training data
            y (np array): classes of x for training data

        Returns:
            self.node_dict(dictionary): dictionary whose key is a node's binary representation (bin_rep)
                                        and value is a Node
        """
        full_mat = append_y_to_atts(x, y)
        self.full_matrix = full_mat

        tree = Node(self.node_dict, self.full_matrix, bin_rep="")
        tree.fit()

        return self.node_dict



    def predict(self, x):
        """
        Predicts the classes for a given attributes matrix X.

        Args:
            x (np array): attributes matrix with K columns where K is the number of columns of training data

        Returns:
            y_hat (np array): predicted class labels for each row of X.
        """
        y_hat = np.apply_along_axis(self.predict_single_row, axis = 1, arr = x)
        return y_hat

    def predict_single_row(self, data_row):
        """
        Predicts the classes for a given row of data in X.

        Args:
            data_row (row vector): row of attribute data

        Returns:
            predicted_class: predicted class for the row of data
        """
        bin_rep = ""

        while not self.node_dict[bin_rep].isTerminal():
            # Find column and value to spliy by
            new_split_val = self.node_dict[bin_rep].split_val
            new_col = self.node_dict[bin_rep].att_col

            # Iterate down the tree
            if data_row[new_col] < new_split_val:
                bin_rep = bin_rep + '0'
            else:
                bin_rep = bin_rep + '1'

        predicted_class = mode(self.node_dict[bin_rep].full_mat[:, -1]) # majority voting

        return predicted_class


    def cart_from_bin(self, bin_string):
        """
        Generates Cartesian Coordinates of node given binary representation of decision tree node.

           Args:
               self
               bin_string(string): binary string representing position of node on decision tree.

           Returns:
               coords(list): list of co-ordinates representing each position of each node on image
               Text(list): list of annotations for each node on image.
           """
        x = 0
        depth = -len(bin_string)
        for i in range(len(bin_string)):
            if bin_string[i] == '0':
                x -= (100 * (0.5 ** i))
            else:
                x += (100 * (0.5 ** i))
        return (x,depth)


    def coords_for_plot(self):
        """
        Generate co-ordinates for image plot of decision tree.

           Args:
               self

           Returns:
               coords(list): list of co-ordinates representing each position of each node on image
               Text(list): list of annotations for each node on image.
           """
        key_list = list(self.node_dict.keys())
        text = []
        coords = []
        ann_coords = []
        for i in range(len(key_list)):
            if key_list[i] == '':
                coordinate = self.cart_from_bin(key_list[i])
                coords.append(coordinate)
                entropy = str(self.node_dict[key_list[i]].entropy)[0:5]
                split_val = str(self.node_dict[key_list[i]].split_val)[0:4]
                column = self.node_dict[key_list[i]].att_col
                new_text = f"Entropy: {entropy}\nSplit val: {split_val}\nColumn: {column}"
                text.append(new_text)
                ann_coords.append(coordinate)
            else:
                coordinate = self.cart_from_bin(key_list[i])
                parent = key_list[i][0:len(key_list[i])-1]
                parent_coordinate = self.cart_from_bin(parent)
                ann_coord = self.cart_from_bin(key_list[i])
                coords.append([[parent_coordinate[0], coordinate[0]],
                               [parent_coordinate[1], coordinate[1]]])
                entropy = str(self.node_dict[key_list[i]].entropy)[0:5]
                split_val = str(self.node_dict[key_list[i]].split_val)[0:3]
                column = self.node_dict[key_list[i]].att_col
                new_text = f"Entropy: {entropy}\nSplit val: {split_val}\nColumn: {column}"
                text.append(new_text)
                ann_coords.append(ann_coord)
        return coords, text, ann_coords


    def plot_tree(self, annotate = False):
        """
        Plots decision tree as an image.

           Args:
               self
               fig_filename(str): Filename under which figure is saved.
               annotate(boolean): Turns annotation on plot on or off

           Returns:
               predicted_class: predicted class for the row of data
           """
        plt.figure(figsize = (9,12))
        plotted = set()
        coords, text, ann_cord = self.coords_for_plot()
        for i in range(len(coords)):
            plt.plot(coords[i][0], coords[i][1])     
                    
            if annotate:
                try:
                    if (ann_cord[i][1], ann_cord[i][0]) not in plotted:
                        plt.annotate(text[i], (ann_cord[i][0], ann_cord[i][1]), fontsize = 7)
                        plotted.add((ann_cord[i][1], ann_cord[i][0]))
                except (TypeError, IndexError):
                    pass
                    plt.annotate(text[i], (ann_cord[i][1], ann_cord[i][0]), fontsize = 0.1)
  
        plt.show()

    def prune(self, x):
        nodes = list(self.node_dict.keys())
        prune_set = set()
        for node in nodes:
            if self.node_dict[node].left_node.isTerminal() and self.node_dict[node].right_node.isTerminal():
                prune_set.add(node)
        prune_list = list(prune_set)
        for node in prune_list:
            pred_orig = self.predict(x)
            self.node_dict[node].isTerminal = True
            pred_new = self.predict(x)
            if pred_orig > pred_new:
                self.node_dict[node].isTerminal = False
        return self.node_dict


def mode(x):
    """
    Calculates the mode of a list or array.

    Args:
        x (list/array)

    Returns:
        mode of X
    """
    vals, counts = np.unique(x, return_counts=True)
    index = np.argmax(counts)
    return vals[index]
