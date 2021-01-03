from seqeval.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report
)
import numpy as np
from typing import List, Dict
# https://huggingface.co/metrics/seqeval
# https://github.com/huggingface/transformers/blob/master/examples/token-classification/run_ner.py
# https://github.com/chakki-works/seqeval
# https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
#  Metrics


def compute_metrics(p, label_list: List[str]) -> Dict:
    labels, predictions = p
    predictions = np.argmax(predictions, axis=-1)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

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
        ['O',          'O',        'O',         'B-MISC',     'I-MISC',    'I-MISC',   'I-MISC',    'O',        'O'],
        ['O',          'B-PER',    'I-PER',     'O',          'O',         'B-MISC',   'I-MISC',    'I-MISC',   'O']
    ]
    y_pred_np = np.array([
        # 'O',         'O',        'O',         'O',         'B-MISC',    'I-MISC',    'I-MISC',    'O',         'O'
        [[10,2,2,1,2],[10,2,2,1,2],[10,1,2,1,2],[10,1,1,2,1],[2,10,1,2,2],[1,1,10,2,1],[1,1,10,2,1],[10,1,1,2,1],[10,2,2,1,2]],
        #'O',         'B-PER',     'I-PER',     'O',         'O',         'B-MISC',    'O',         'I-MISC',         'O'
        [[10,2,2,1,2],[1,2,2,10,2],[1,2,2,1,10],[10,2,2,1,2],[10,2,2,1,2],[1,10,2,1,2],[1,2,10,1,2],[1,2,10,1,2],[10,2,2,1,2]]
    ])
    # codes        0    1         2         3        4
    label_list = ['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER']
    m = compute_metrics((y_true_np, y_pred_np), label_list)
    for k, v in m.items():
        print(k, v)
    report = classification_report(y_true, y_pred)
    print(report)


if __name__ == "__main__":
    self_test()
