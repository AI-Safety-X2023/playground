from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from .nn import Hyperparameters

def labels_and_predictions(
        model: nn.Module,
        dataset: Dataset,
        as_logits=False,
        hp=Hyperparameters(),
    ):
    """
    Returns the dataset labels `y_true` and the model predictions `y_pred`.

    If `as_logits` is `False`, `y_pred` contains the predicted classes.
    Otherwise, `y_pred` contains the logits (unnormalized prediction values for each class).
    """
    model.eval()
    
    # FIXME: we should not assume that the dataset has such a field
    y_true = dataset.targets

    loader = DataLoader(dataset, batch_size=hp.inference_batch_size)

    # Perform inference on batches and collect results in a single tensor
    # TODO: use torch.stack? problem with last batch which has different length
    with torch.no_grad():
        if not as_logits:
            y_pred = torch.zeros(len(dataset))
        else:
            Xb, _ = next(iter(loader))
            y = model(Xb[0:1])
            num_classes = y.shape[1]

            y_pred = torch.zeros((len(dataset), num_classes))
        
        i = 0
        for X, _ in loader:
            with torch.autocast(device_type=X.device.type):
                y_p = model(X).detach()
            if not as_logits:
                y_p = y_p.argmax(1)
            y_pred[i : i + len(y_p)] = y_p
            i += len(X)
    
    return y_true, y_pred

def most_frequent_confusions(y_true: Tensor, y_pred: Tensor) -> np.ndarray:
    """
    Returns the most frequent confusion for each class.

    # Parameters
    - y_pred:  a batch of predicted labels.
    
    # Returns
    confusions: an array of same length as the number of classes.
    
    For each class `c1`, `confusions[c1]` is the most frequent other class
    that is predicted when the true class is `c1`.
    """
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)
    return cm.argmax(axis=-1)

def _most_frequent_second_predictions(y_logits: Tensor) -> np.ndarray:
    """
    Returns the most frequent second prediction after each class.

    # Parameters
    - y_logits: a batch of class logits

    # Returns
    - seconds: an array of same length as the number of classes.
    For each class `c1`, `seconds[c1]` is the most frequent class `c2`
    such that the vector of top-2 predictions is `[c1, c2]`.
    """
    # Get the number of classes
    C = y_logits.shape[-1]

    # Get the top 2 predictions for each sample
    #np.argpartition(pred, kth=-2, axis=-1)[:, -2:]
    top2_preds = torch.topk(y_logits, k=2, axis=-1).indices

    # Encode top 2 predictions as a decimal number
    top2_preds_enc = C * top2_preds[:, 0] + top2_preds[:, 1]

    # Count the number of occurrences of each "top 2 prediction"
    pairs, counts = np.unique(top2_preds_enc, return_counts=True)

    # Sort by number of occurrences
    #count_sort_idx = np.argsort(-counts)
    #pairs = pairs[count_sort_idx]

    seconds = np.arange(C)
    seconds_counts = np.zeros(C, int)
    for (pair, count) in zip(pairs, counts):
        # Decode pair
        first, second = pair // C, pair % C
        if count > seconds_counts[first]:
            seconds[first] = second
            seconds_counts[first] = count
    return seconds