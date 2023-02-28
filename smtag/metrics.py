import pdb
from typing import List, Dict
from smtag.train.train_seq2seq import HfSeq2SeqTrainer
from transformers import EvalPrediction
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from seqeval.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report
)
import re

def compute_metrics_lm(eval_pred: EvalPrediction):
    """Compute recall at the masked position
    """
    predictions, labels = eval_pred
    mask = labels != -100
    # filter everything except the masked position and flatten tensors
    labels = labels[mask].flatten()
    predictions = np.argmax(predictions, axis=-1)
    predictions = predictions[mask].flatten()
    _, recall, _, _ = precision_recall_fscore_support(y_true=labels, y_pred=predictions, average='micro')
    return {'recall': recall}


class MetricsTOKCL:
    """Computes metrics for token classifications. Assumes the labels follow the IOB2 scheme.

    Args:
        label_list: the list of IOB2 string labels.
    """
    def __init__(self, label_list: List = []):
        self.label_list = label_list

    def __call__(self, eval_pred: EvalPrediction) -> Dict:
        """Computes accuracy precision, recall and f1 based on the list of IOB2 labels. 
        Positions with labels with a value of -100 will be filtered out both from true labela dn prediction.

        Args:
            eval_pred (EvalPrediction): the predictions and targets to be matched as np.ndarrays.

        Returns:
            (Dict): a dictionary with accuracy_score, precision, recall and f1.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        print("\n"+" " * 80)
        try:
            print(classification_report(true_labels, true_predictions))
        except ValueError as e:
            print(e)
            import pdb; pdb.set_trace()
        return {
            "accuracy_score": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }


class MetricsNerSeq2seq(HfSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        pass