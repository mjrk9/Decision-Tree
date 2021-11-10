import numpy as np

# Classification Metrics

multi_confusion_matrix = np.array([[0, 0, 0, 0],
                            [1, 2, 3, 4],
                            [2, 3, 4, 5],
                            [3, 4, 8, 9]])

# Each row represents the actual class labels (1, 2, 3, 4)
# Each column represents the predicted class label

# Define one class as positive and the others as negative

def convert_to_binary_confusion_matrix(confusion_matrix):
    """ Convert a multi-class confusion matrix into K binary confusion matrices.

    K is the number of classes. For each matrix, one class is labelled as positive,
    and the remaining K-1 classes are labelled as negative.
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

# print(convert_to_binary_confusion_matrix(confusion_matrix= multi_confusion_matrix))

def recall(confusion_matrix):
    """ Gives the recall of a 2 X 2 confusion matrix.

    Recall is the total number of correctly classified positive examples (True Positives),
    divided by the total number of positive examples (TP + FN)
    """

    TP = confusion_matrix[0,0]
    FN = confusion_matrix[0,1]

    return (TP/ (TP + FN))


def precision(confusion_matrix):
    """ Gives the precision of a 2 X 2 confusion matrix.

    Precision is the total number of correctly classified positive examples (True Positives),
    divided by the total number of predicted positive examples (TP + FP).
    """

    TP = confusion_matrix[0, 0]
    FP = confusion_matrix[1, 1]
    return TP / (TP + FP)

def f1(precision, recall):
    """ Gives the f1 score given precision and recall.

    f1 = (2 * precision * recall) / (precision + recall)

    Args:
        precision(float)
        recall(float)

    Returns:
        f1 (float)
    """
    return (2 * precision * recall) / (precision + recall)

x = np.array([[3, 1],
              [2, 2]])

# print(recall(x), precision(x), f1(precision(x), recall(x)))

def calc_metrics_each_class(multi_confusion_matrix):
    metrics_array = []
    for matrix in convert_to_binary_confusion_matrix(multi_confusion_matrix):
        print("matrix", matrix)
        recall_value = recall(matrix)
        precision_value = precision(matrix)
        f1_score = f1(precision_value, recall_value)
        metrics_array.append([recall_value, precision_value, f1_score])

calc_metrics_each_class(multi_confusion_matrix)


