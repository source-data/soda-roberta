from transformers import EvalPrediction
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


def compute_metrics(pred: EvalPrediction):
    """Compute recall at the masked position
    """
    mask = pred.label_ids != -100
    # filter everything except the masked position and flatten tensors
    labels = pred.label_ids[mask].flatten()
    preds = pred.predictions[mask].flatten()
    _, recall, _, _ = precision_recall_fscore_support(y_true=labels, y_pred=preds, average='micro')
    return {'recall': recall}


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
            [   2,    6,    8],  # 1 false positive, irrelevant pos will be ignored
            [   1,    7,    4]   # 1 true positive, irrelevant pos will be ignored
        ]) 
    )
    m = compute_metrics(pred)
    print(f"recall={m['recall']}")
    assert m['recall'] == 0.75
    print("Looks like it is working!")


if __name__ == "__main__":
    self_test()
