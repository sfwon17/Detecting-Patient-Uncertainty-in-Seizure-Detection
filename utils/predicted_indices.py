def get_indices(predictions, labels, pred_label, true_label):
    """
    Get indices where the predictions match a specified value and 
    the labels match a specified value.
    
    Args:
        predictions (list): List of predicted labels.
        labels (list): List of true labels.
        pred_label (int): The prediction label to match.
        true_label (int): The true label to match.
    
    Returns:
        list: Indices where the conditions are met.
    """
    indices = [i for i in range(len(predictions)) if predictions[i] == pred_label and labels[i] == true_label]
    return indices
