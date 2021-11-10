from scripts.Tree import Tree
from scripts.entropy import *
from scripts.evaluation_metrics import *
from scripts.cross_validation import *
import matplotlib.pyplot as plt

filepath_clean = 'wifi_db/clean_dataset.txt'
filepath_noisy = 'wifi_db/noisy_dataset.txt'

x, y, classes = read_dataset(filepath_clean)

# Train and test on full dataset
tree = Tree()
node_dict = tree.fit(x, y)


#  Cross validation
def train_tree_CV(x, y):
    """
           This method trains the decision tree.

           Args:
               x (numpy.ndarray) : the x train set (data)
               y (numpy.ndarray): the y train set (labels)

           Returns:
               CV_accuracy (float): the accuracy metric
               CV_recall  (float): the recall metric
               CV_precision (float): the precision metric
               CV_f1 (float) : the f1-score metric
               CV_full_conf_mats (numpy.ndarray) : the full confusion matrix
           """

    CV_data = generate_CV_data(x, y)

    metrics_each_fold = []
    full_conf_mats = []

    for fold in CV_data:
        train_data = fold[0]
        test_data = fold[1]

        x_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        x_test = test_data[:, :-1]
        y_test = test_data[:, -1]

        # Train decision tree
        tree = Tree()
        tree.fit(x_train, y_train)
        y_hat = tree.predict(x_test)
        multiclass_confusion_matrix = confusion_matrix(y_test, y_hat)
        full_conf_mats.append(multiclass_confusion_matrix)
        accuracy = calc_accuracy(multiclass_confusion_matrix)
        macro_recall, macro_precision, macro_f1 = calc_macro_scores(multiclass_confusion_matrix)

        metrics_each_fold.append([accuracy, macro_recall, macro_precision, macro_f1])

    metrics_each_fold = np.array(metrics_each_fold)
    CV_accuracy = np.average(metrics_each_fold[:, 0])
    CV_recall = np.average(metrics_each_fold[:, 1])
    CV_precision = np.average(metrics_each_fold[:, 2])
    CV_f1 = np.average(metrics_each_fold[:, 3])

    return CV_accuracy, CV_recall, CV_precision, CV_f1, full_conf_mats



#Tree pruning
def train_tree_pruned(x, y, unpruned_accuracy):
    """
       This method prunes the decision tree.

       For a given node in the tree the method obtains the children. If the children are terminal (leaf nodes),
       the children are disabled and the given node becomes a terminal node.

       If this change leads to an improvement in accuracy the node is permanently pruned. If this changes does not
       improve accuracy, this node is unpruned and its children are enabled again.


       Args:
           x (numpy.ndarray) : the x train set (data)
           y (numpy.ndarray): the y train set (labels)

       Returns:
           accuracy (float)
           recall  (float)
           precision (float)
           f1 (float) : f1-score
           full_conf_mats (numpy.ndarray) : the full confusion matrix
       """

    CV_data = generate_CV_data(x, y)
    metrics_each_fold = []
    full_conf_mats=[]
    for fold in CV_data:
        train_data = fold[0]
        test_data = fold[1]
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        x_test = test_data[:, :-1]
        y_test = test_data[:, -1]
       ###
        # Train decision tree
        tree = Tree()
        Tree_to_prune = tree.fit(x_train, y_train)

        Nodes_Benefical_prune = {}

        ##find nodes with two leaves
        current_accuracy = unpruned_accuracy
        Tree_temp = dict(Tree_to_prune)

        for i in Tree_to_prune:
            left_node = Tree_to_prune[i].left_node
            try:
                left_node_terminal = left_node.isTerminal()
            except:
                left_node_terminal = False
            right_node = Tree_to_prune[i].right_node
            try:
                right_node_terminal = right_node.isTerminal()
            except:
                right_node_terminal = False
            if left_node_terminal == True & right_node_terminal == True:

                Tree_temp[i].isPruned = True

                d = Tree(Tree_temp)
                y_hat = d.predict(x_test)
                multiclass_confusion_matrix = confusion_matrix(y_test, y_hat)
                accuracy = calc_accuracy(multiclass_confusion_matrix)

                if accuracy >= current_accuracy:
                    current_accuracy = accuracy

                if current_accuracy > accuracy:
                    Tree_temp[i].isPruned = False

        # multiclass_confusion_matrix = confusion_matrix(y_test, y_hat)
        full_conf_mats.append(multiclass_confusion_matrix)
        accuracy = calc_accuracy(multiclass_confusion_matrix)
        macro_recall, macro_precision, macro_f1 = calc_macro_scores(multiclass_confusion_matrix)
        metrics_each_fold.append([accuracy, macro_recall, macro_precision, macro_f1])

    metrics_each_fold = np.array(metrics_each_fold)

    CV_accuracy = np.average(metrics_each_fold[:, 0])
    CV_recall = np.average(metrics_each_fold[:, 1])
    CV_precision = np.average(metrics_each_fold[:, 2])
    CV_f1 = np.average(metrics_each_fold[:, 3])

    return CV_accuracy, CV_recall, CV_precision, CV_f1, full_conf_mats

    print(metrics_each_fold)
    accuracy = np.average(metrics_each_fold[:, 0])
    recall = np.average(metrics_each_fold[:, 1])
    precision = np.average(metrics_each_fold[:, 2])
    f1 = np.average(metrics_each_fold[:, 3])

    print(accuracy, recall, precision, f1)
    return accuracy, recall, precision, f1, full_conf_mats


# CV_accuracy, CV_recall, CV_precision, CV_f1, con_mats = train_tree_CV(x, y)
# Pruned_CV_accuracy, Pruned_CV_recall, Pruned_CV_precision, Pruned_CV_f1,Puned_con_mats = train_tree_pruned(x, y, CV_accuracy)

# full = np.sum(Puned_con_mats, axis=0)
# full_mean = np.mean(Puned_con_mats, axis = 0)

# print("Noisy:")
# print("PRUNE:")
# print("full prune: ", full)
# print("mean prune: ", full_mean)
# for i in range(len(full_mean)):
#     print("Recalls: ", full_mean[i][i] / np.sum(full_mean[i]))
# for i in range(len(full_mean)):
#     print("Precisions: ", full_mean[i][i] / np.sum(full_mean[:,i]))

print("Not prune:")
full_mean_pre = np.mean(con_mats, axis = 0)
print(full_mean_pre)
for i in range(len(full_mean_pre)):
    print("Recalls: ", full_mean_pre[i][i] / np.sum(full_mean_pre[i]))
for i in range(len(full_mean_pre)):
    print("Precisions: ", full_mean_pre[i][i] / np.sum(full_mean_pre[:,i]))











