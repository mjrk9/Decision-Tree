import numpy as np

def confusion_matrix(y_true, y_prediction, class_labels=None):
    """ Compute the confusion matrix.

    Args:
        y_true (np.ndarray): the correct ground truth labels
        y_prediction (np.ndarray): the predicted labels
        class_labels (np.ndarray): a list of unique class labels.
                               Defaults to the union of y_true and y_prediction.

    Returns:
        np.array : shape (C, C), where C is the number of classes.
                   Rows are ground truth per class, columns are predictions
    """

    # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_true, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

    # for each correct class (row),
    # compute how many instances are predicted for each class (columns)
    for (i, label) in enumerate(class_labels):
        # get predictions where the ground truth is the current class label
        indices = (y_true == label)
        true = y_true[indices]
        predictions = y_prediction[indices]

        # quick way to get the counts per label
        (unique_labels, counts) = np.unique(predictions, return_counts=True)

        # convert the counts to a dictionary
        frequency_dict = dict(zip(unique_labels, counts))

        # fill up the confusion matrix for the current row
        for (j, class_label) in enumerate(class_labels):
            confusion[i, j] = frequency_dict.get(class_label, 0)

    return confusion


def convert_to_binary_confusion_matrix(confusion_matrix):
    """ Convert a multi-class confusion matrix into C binary confusion matrices.

    C is the number of classes. For each matrix, one class is labelled as positive,
    and the remaining C-1 classes are labelled as negative.

    Args:
        confusion_matrix(np array): shape (C, C), where C is the number of classes.
                Rows are ground truth per class, columns are predictions

    Returns:
        C binary confusion matrices (list of np array): each confusion matrix is of shape
                                                        2 by 2
    """

    classes = len(confusion_matrix[0])

    binary_confusion_matrices = []
    for k in range(classes):
        TP = confusion_matrix[k, k]
        FP = np.sum(confusion_matrix[:, k]) - confusion_matrix[k, k]
        FN = np.sum(confusion_matrix[k, :]) - confusion_matrix[k, k]
        TN = np.sum(confusion_matrix) - TP - FN - FP

        new_confusion_matrix = np.array([[TP, FN],
                                        [FP, TN]])
        binary_confusion_matrices.append(new_confusion_matrix)

    return binary_confusion_matrices


def calc_accuracy(confusion_matrix):
    """ Gives the accuracy of a confusion matrix.

    Accuracy is the number of correctly classified examples divided by the total number of examples.

    Args:
        confusion_matrix(array) : a C X C confusion matrix where C refers to the number of classes.

    Returns:
        accuracy (float)
    """

    correct_classifications = np.sum(np.diagonal(confusion_matrix))
    total_examples = np.sum(confusion_matrix)

    accuracy = correct_classifications/total_examples
    return accuracy


def calc_recall(confusion_matrix):
    """ Gives the recall of a 2 X 2 confusion matrix.

    Recall is the total number of correctly classified positive examples (True Positives),
    divided by the total number of positive examples (TP + FN)

    Args:
        confusion_matrix(array) : a 2 X 2 confusion matrix

    Returns:
        recall (float)
    """

    TP = float(confusion_matrix[0,0])
    FN = float(confusion_matrix[0,1])

    return (TP/ (TP + FN))


def calc_precision(confusion_matrix):
    """ Gives the precision of a 2 X 2 confusion matrix.

    Precision is the total number of correctly classified positive examples (True Positives),
    divided by the total number of predicted positive examples (TP + FP).

    Args:
        confusion_matrix(array) : a 2 X 2 confusion matrix

    Returns:
        precision (float)
    """

    TP = confusion_matrix[0, 0]
    FP = confusion_matrix[1, 0]
    return TP / (TP + FP)


def calc_f1(confusion_matrix):
    """ Gives the f1 score given precision and recall.

    f1 = (2 * precision * recall) / (precision + recall)

    Args:
        confusion_matrix(array) : a 2 X 2 confusion matrix

    Returns:
        f1 (float)
    """
    precision = calc_precision(confusion_matrix)
    recall = calc_recall(confusion_matrix)

    return (2 * precision * recall) / (precision + recall)


def calc_metrics_each_class(multi_confusion_matrix):
    """ Gives the metrics for each class.

        Args:
            multi_confusion_matrix(np.ndarray) : contains multiple confusion matrix's

        Returns:
            metrics_array(list): a list of metrics
        """



    metrics_array = []
    for matrix in convert_to_binary_confusion_matrix(multi_confusion_matrix):
        recall_value = calc_recall(matrix)
        precision_value = calc_precision(matrix)
        f1_score = calc_f1(matrix)
        metrics_array.append([recall_value, precision_value, f1_score])

    metrics_array = np.array(metrics_array)

    return metrics_array


def calc_macro_scores(multi_confusion_matrix):
    """ Returns multiple metrics at once including recall, precision and f1-score

            Args:
                multi_confusion_matrix(np.ndarray) : contains a multiclass confusion matrix

            Returns:
                metrics_array(list): a list of metrics
            """

    class_scores = calc_metrics_each_class(multi_confusion_matrix)
    macro_recall = np.average(class_scores[:, 0])
    macro_precision = np.average(class_scores[:, 1])
    macro_f1 = np.average(class_scores[:, 2])

    return macro_recall, macro_precision, macro_f1






