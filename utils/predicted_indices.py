# Get indices where the predictions match a specified value and the labels match a specified value.
def get_indices(predictions, labels, pred_label, true_label):
    indices = [i for i in range(len(predictions)) if predictions[i] == pred_label and labels[i] == true_label]
    return indices
