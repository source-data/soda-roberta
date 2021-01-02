from transformers import EvalPrediction
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.flatten()
    # need to ignore the padding labels -100
    set_of_labels = set(labels)
    set_of_labels.remove(-100)
    set_of_labels = list(set_of_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels, y_pred=preds, labels=set_of_labels, average='micro')
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def self_test():
    pred = EvalPrediction(
        label_ids=np.array([
            [-100,    1, -100],
            [   2, -100, -100],
            [-100, -100,    3],
            [-100, -100,    4]
        ]),
        predictions=np.array([
            [-100,    1, -100],  # 1 true positive
            [   2, -100, -100],  # 1 true positive
            [   1,    1,    1],  # 3 false negatives
            [   1,    1,    1]   # 3 false negatives
        ])  # => 2 TP / 8 calls = 0.25 precision; 2 TP / 4 P = 50%  recall
    )
    metrics = compute_metrics(pred)
    print(metrics)
    assert metrics['precision'] == 0.25
    assert metrics['recall'] == 0.50
    print("Looks like it is working!")


if __name__ == "__main__":
    self_test()
