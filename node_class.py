from entropy import *
import matplotlib.pyplot as plt


class Node:

    def __init__(self, node_dict, full_matrix , bin_rep = '0'):
        self.full_mat = full_matrix
        self.bin_rep = bin_rep
        self.node_dict = node_dict
        self.split_val = 'N/A'
        self.att_col = 'N/A'

        self.left = None
        self.right = None
        self.left_node = None
        self.right_node = None

        classes, y = np.unique(full_matrix[:,-1], return_inverse=True)

        # print("classes:" ,classes)
        # print("y", y)
        self.entropy = entropy(full_matrix[:,-1], classes)

        if not self.isTerminal():
            self.left, self.right, self.att_col, self.split_val = self.split_to_children()

    def fit(self):

        if self.isTerminal():
            self.node_dict[self.bin_rep] = self

        else:
            # Place the left as self.left_node
            self.left_node = Node(self.node_dict, self.left, bin_rep = self.bin_rep+'0')
            self.left_node.fit()

            self.right_node = Node(self.node_dict, self.right, bin_rep = self.bin_rep+'1')
            self.right_node.fit()

            self.node_dict[self.bin_rep] = self


    def split_to_children(self):
        """
        Takes the current node's full matrix and splits it according to the optimal column and split value
        """
        self.att_col, self.split_val = self.find_split()
        _, left, right = calc_info_gain_of_matrix(self.full_mat, self.att_col, self.split_val)

        self.left = left
        self.right = right
        return self.left, self.right, self.att_col, self.split_val


    def isTerminal(self):
        """
        Checks if the current node is terminal (i.e. if the leaves are pure)
        """
        y_class, _ = np.unique(self.full_mat[:,-1], return_inverse=True)
        return (abs(entropy(self.full_mat[:-1], y_class)) == 0)


    def find_split(self):
        """
        Chooses the best column and best split value for splitting
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


class Tree():

    def __init__(self, filepath, node_dict = {}):
        x, y, classes = read_dataset(filepath)
        full_mat = append_y_to_atts(x, y)
        self.node_dict = node_dict
        self.full_matrix = full_mat

    def fit(self, bin_string = ''):
        tree = Node(self.node_dict, self.full_matrix, bin_string)
        tree.fit()
        return self.node_dict


    def cart_from_bin(self, bin_string):
        x = 0
        depth = -len(bin_string)
        for i in range(len(bin_string)):
            if bin_string[i] == '0':
                x -= (100 * (0.5 ** i))
            else:
                x += (100 * (0.5 ** i))
        return (x,depth)


    def coords_for_plot(self):
        key_list = list(self.node_dict.keys())
        text = []
        coords = []
        for i in range(len(key_list)):
            if key_list[i] == '':
                coordinate = self.cart_from_bin(key_list[i])
                coords.append(coordinate)
                entropy = self.node_dict[key_list[i]].entropy
                split_val = self.node_dict[key_list[i]].split_val
                column = self.node_dict[key_list[i]].att_col
                new_text = f"Entropy: {entropy}\nSplit val: {split_val}\nColumn: {column}"
                text.append(new_text)
            else:
                coordinate = self.cart_from_bin(key_list[i])
                parent = key_list[i][0:len(key_list[i])-1]
                parent_coordinate = self.cart_from_bin(parent)
                coords.append([[parent_coordinate[0], coordinate[0]],
                               [parent_coordinate[1], coordinate[1]]])
                entropy = self.node_dict[key_list[i]].entropy
                split_val = self.node_dict[key_list[i]].split_val
                column = self.node_dict[key_list[i]].att_col
                new_text = f"Entropy: {entropy}\nSplit val: {split_val}\nColumn: {column}"
                text.append(new_text)
        return coords, text


    def plot_tree(self, fig_filename, annotate = False):
        coords, text = self.coords_for_plot()
        for i in range(len(coords)):
            plt.plot(coords[i][0], coords[i][1])
            if annotate:
                try:
                    plt.annotate(text[i], (coords[i][0][0], coords[i][1][0]))
                except TypeError:
                    plt.annotate(text[i], (coords[i][0], coords[i][1]), fontsize = 0.1)
            plt.savefig(fig_filename)


    def single_data_test_labelled(self, data_row, first_node_bin_rep = ''):
        bin_rep = first_node_bin_rep
        while not self.node_dict[bin_rep].isTerminal():
            new_split_val = self.node_dict[bin_rep].split_val
            new_col = self.node_dict[bin_rep].att_col
            if data_row[new_col] < new_split_val:
                bin_rep = bin_rep + '0'
            else:
                bin_rep = bin_rep + '1'

        comparison_class = self.node_dict[bin_rep].full_mat[-1][-1]
        if int(data_row[-1]) == comparison_class:
            return 1
        else:
            return 0


    def test_labelled_data(self, filepath):
        x_test, y_test, labels = read_dataset(filepath)
        full_mat_test = append_y_to_atts(x_test, y_test)
        correct = 0
        total = 0
        for row in range((full_mat_test.shape[0])):

            correct += self.single_data_test_labelled(full_mat_test[row])
            total += 1
        return correct/total
