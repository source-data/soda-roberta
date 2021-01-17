from seqeval.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report
)
from transformers import EvalPrediction
import numpy as np
from typing import List, Dict, Tuple
# https://huggingface.co/metrics/seqeval
# https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
# https://github.com/chakki-works/seqeval
# https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
#  Metrics


class MetricsComputer:
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
        print(classification_report(true_labels, true_predictions))
        return {
            "accuracy_score": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }


def self_test():
    y_true = [
        ['O',          'O',         'O',          'O',        'B-MISC',   'I-MISC',   'I-MISC',       'O',     'O'],
        ['O',          'B-PER',    'I-PER',       'I-PER',    'O',        'B-MISC',   'I-MISC',       'O',     'O']
    ]
    y_true_np = np.array([
        # 'O',         'O',        'O',          'O',         'B-MISC',    'I-MISC',   'I-MISC',     'O',      'O'
        [-100,          0,          0,            0,           1,           2,          2,            0,       -100],
        #'O',          'B-PER',     'I-PER',     'O',         'O',         'B-MISC',   'I-MISC',     'O',      'O'
        [-100,          3,          4,            0,           0,           1,          2,            0,        -100]
    ])
    y_pred = [
        ['O',          'O',        'B-PER',         'O',      'B-MISC',     'I-MISC',   'I-MISC',    'O',        'O'],
        ['O',          'B-PER',    'I-PER',     'I-PER',      'O',          'B-MISC',   'O',         'B-MISC',   'O']
    ]
    y_pred_np = np.array([
        # 'O',         'O',        'O',         'O',         'B-MISC',    'I-MISC',    'I-MISC',    'O',         'O'
        [[10,2,2,1,2],[10,2,2,1,2],[10,1,2,1,2],[10,1,1,2,1],[2,10,1,2,2],[1,1,10,2,1],[1,1,10,2,1],[10,1,1,2,1],[10,2,2,1,2]],
        #'O',         'B-PER',     'I-PER',     'O',         'O',         'B-MISC',    'O',         'B-MISC',         'O'
        [[10,2,2,1,2],[1,2,2,10,2],[1,2,2,1,10],[10,2,2,1,2],[10,2,2,1,2],[1,10,2,1,2],[10,2,1,1,2],[1,10,1,1,2],[10,2,2,1,2]]
    ])
    
    mc = MetricsComputer(label_list=[
    #    0    1         2         3        4
        'O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER'
    ])
    eval_pred = EvalPrediction(y_pred_np, y_true_np)
    m = mc(eval_pred)
    print(m)
    # for k, v in m.items():
    #     print(k, v)
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    self_test()
