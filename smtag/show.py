from random import randrange, sample
from typing import Dict

import torch
from transformers import RobertaTokenizerFast, TrainerCallback

from .config import config


# uses spcial color characters for the console output
# for code in {1..256}; do printf "\e[38;5;${code}m"$code"\e[0m";echo; done
# for i = 1, 32 do COLORS[i] = "\27[38;5;"..(8*i-7).."m" end
# printf "\e[30;1mTesting color\e[0m"
# for i in range(25,50): print(f"\033[{i};1mTesting color {i}\033[0m")
# for i in range(256): print(f"\033[38;5;{i}mBlahblah color={i}\033[0mAnd normal")


class ShowExample(TrainerCallback):
    """Visualizes on the console the result of a prediction with the current state of the model.
    It uses a randomly picked input example and decodes input and output with the provided tokenizer.
    The predicted words are colored depending on whether the prediction is correct or not.
    If the prediction is incorrect, the expected word is displayed in square brackets.

    Args:

        tokenizer (RobertaTokenizer): the tokenizer used to generate the dataset.

    Class Attributes:

        COLOR_CHAR (Dict): terminal colors used to produced colored string
    """

    COLOR_CHAR = {}

    def __init__(self, tokenizer: RobertaTokenizerFast, *args, **kwargs):
        self.tokenizer = tokenizer

    def on_evaluate(
        self,
        *args,
        model=None,
        eval_dataloader: torch.utils.data.DataLoader = None,
        **kwargs
    ):
        """Method called when evaluating the model. Only the needed kwargs are unpacked.

        Args:

            model: the current model being trained.
            eval_dataloader (torch.utils.data.DataLoader): the DataLoader used to produce the evaluation examples
        """
        with torch.no_grad():
            inputs = self.pick_random_example(eval_dataloader)
            pred = model(inputs["input_ids"], labels=inputs["labels"], attention_mask=inputs["attention_mask"])
            pred_idx = pred['logits'].argmax(-1)[0].cpu()
        inputs = {k: v[0] for k, v in inputs.items()}
        self.to_console(inputs, pred_idx)

    def pick_random_example(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        L = len(dataloader.dataset)
        dataset = dataloader.dataset
        rand_example_idx = randrange(L)
        batch = dataloader.collate_fn([dataset[rand_example_idx]])  # batch with a single random example
        inputs = {}
        for k, v in batch.items():
            inputs[k] = v.cuda() if torch.cuda.is_available() else v
        return inputs

    def to_console(self, inputs: Dict[str, torch.Tensor], pred_idx):
        pred_idx = [e.item() for e in pred_idx]
        input_ids = [e.item() for e in inputs["input_ids"]]
        labels = [e.item() for e in inputs["labels"]]
        colored = ""
        for i in range(len(input_ids)):
            input_id = input_ids[i]
            label_idx = pred_idx[i]
            true_label = labels[i]
            colored += self._correct_incorrect(input_id, label_idx, true_label) + " "
        print(f"\n\n{colored}\n\n")

    def _correct_incorrect(self, input_id: int, label_idx: int, true_label: int) -> str:
        raise NotImplementedError


class ShowExampleLM(ShowExample):

    COLOR_CHAR = {
            "blue": '\033[32;1m',
            "red": '\033[31;1m',
            "close": '\033[0m'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_evaluate(self, *args, model=None, eval_dataloader=None, **kwargs):
        with torch.no_grad():
            inputs = self.pick_random_example(eval_dataloader)
            pred = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
            pred_idx = pred['logits'].argmax(-1)[0].cpu()
        inputs = {k: v[0] for k, v in inputs.items()}
        self.to_console(inputs, pred_idx)

    def _correct_incorrect(self, input_id, label_idx, true_label, attention_mask=None) -> str:
        colored = ""
        is_prediction = true_label != -100
        if is_prediction:
            decoded_pred = self.tokenizer.decode(label_idx)
            decoded_label = self.tokenizer.decode(true_label)
            correct = (label_idx == true_label)
            color = "blue" if correct else "red"
            insert = decoded_pred if correct else f"{decoded_pred}[{decoded_label.strip()}]"
            colored = f"{self.COLOR_CHAR[color]}{insert}{self.COLOR_CHAR['close']}"
        elif attention_mask == 1 and input_id is not None:
            colored += self.tokenizer.decode(input_id)
        return colored


class ShowExampleTextGeneration(ShowExampleLM):

    COLOR_CHAR = {
            "blue": '\033[32;1m',
            "red": '\033[31;1m',
            "close": '\033[0m'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = kwargs.get('max_sentence_length', None)
        self.model_config = kwargs.get('model_config', None)
        if not self.max_length:
            self.max_length = config.max_length[1]

    def on_evaluate(self, *args, model=None, eval_dataloader=None, **kwargs):
        with torch.no_grad():
            inputs = self.pick_random_example(eval_dataloader)
            pred_idx = model.generate(inputs["input_ids"], **self.model_config)
        pred_idx = pred_idx[0][1:]  # removing first token that seems to be </s> probably a mistake somewhere
        inputs = {k: v[0] for k, v in inputs.items()}
        self.to_console(inputs, pred_idx)

    def to_console(self, inputs: Dict[str, torch.Tensor], pred_idx):
        pred_idx = [e.item() for e in pred_idx]
        input_ids = [e.item() for e in inputs["input_ids"]]
        labels = [e.item() for e in inputs["labels"]]
        colored = ""
        input_str = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        generated_str = self.tokenizer.decode(pred_idx, skip_special_tokens=True)
        target_no_pad_labels = [l for l in labels if l != -100]
        target_str = self.tokenizer.decode(target_no_pad_labels, skip_special_tokens=True)
        colored += f"input:\n{input_str}\n\n"
        colored += f"generated:\n{generated_str}\n\n"
        colored += f"target:\n{target_str}\n\n"
        colored += "diff:\n"

        for i in range(min(len(labels), len(pred_idx))):
            label_idx = pred_idx[i]
            true_label = labels[i]
            colored += self._correct_incorrect(None, label_idx, true_label)
        print(f"\n\n{colored}\n\n")


class ShowExampleTwinLM(ShowExampleLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_evaluate(self, *args, model=None, eval_dataloader=None, **kwargs):
        with torch.no_grad():
            inputs = self.pick_random_example(eval_dataloader)
            pred = model(**inputs)
            # pred.logits is an array with predictions for twin examples
            for i, pred_logits in enumerate(pred['logits']):
                pred_idx = pred_logits.argmax(-1)[0].cpu()
                # extract input from specific twin example
                inputs_i = {k: v[i][0] for k, v in inputs.items()}
                self.to_console(inputs_i, pred_idx)

    def pick_random_example(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        batch = next(iter(dataloader))
        # batch[_][_] is a list of twin example
        rand_example_idx = randrange(batch['input_ids'][0].size(0))
        inputs = {}
        for k, v in batch.items():
            inputs[k] = []
            for twin in batch[k]:
                ex = twin[rand_example_idx].clone().unsqueeze(0)
                if torch.cuda.is_available():
                    ex = ex.cuda()
                inputs[k].append(ex)
        return inputs


class ShowExampleTOKCL(ShowExample):

    COLOR_CHAR = {
        "underscore": "\033[4m",
        "bold": "\033[1m",
        "close":  "\033[0m",
        "var_color": "\033[38;5;{color_idx}m",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _correct_incorrect(self, input_id, label_idx, true_label, **kwargs) -> str:
        colored = ""
        if input_id != self.tokenizer.pad_token_id:  # don't display padding
            decoded = self.tokenizer.decode(input_id)
            # indicate the true label with underline
            underscore = self.COLOR_CHAR["underscore"] if label_idx == true_label else ''
            if label_idx > 0:  # don't show default no_label
                colored = f"{self.COLOR_CHAR['bold']}{underscore}{self.COLOR_CHAR['var_color'].format(color_idx=label_idx)}{decoded}{self.COLOR_CHAR['close']}"
            else:
                colored = f"{decoded}"
        return colored

class ShowExampleTwinLM(ShowExampleLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_evaluate(self, *args, model=None, eval_dataloader=None, **kwargs):
        with torch.no_grad():
            inputs = self.pick_random_example(eval_dataloader)
            pred = model(**inputs)
            # pred.logits is an array with predictions for twin examples
            for i, pred_logits in enumerate(pred['logits']):
                pred_idx = pred_logits.argmax(-1)[0].cpu()
                # extract input from specific twin example
                inputs_i = {k: v[i][0] for k, v in inputs.items()}
                self.to_console(inputs_i, pred_idx)

    def pick_random_example(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, torch.Tensor]:
        dataset = dataloader.dataset
        L = len(dataset)
        rand_example_idx = randrange(L)
        batch = dataloader.collate_fn([dataset[rand_example_idx]])  # batch with a single random example
        inputs = {}
        for k in batch:
            inputs[k] = [
                twin.cuda() if torch.cuda.is_available() else twin
                for twin in batch[k]
            ]
        return inputs


class ShowExampleCGraphVAEForLM(ShowExampleLM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_evaluate(self, *args, model=None, eval_dataloader=None, **kwargs):
        with torch.no_grad():
            inputs = self.pick_random_example(eval_dataloader)
            pred = model(**inputs)
            pred_idx = pred['logits'].argmax(-1)[0].cpu()
        inputs = {k: v[0] for k, v in inputs.items()}
        if pred.flipped:
            inputs['labels'] = inputs['labels'].flip(0)
        self.to_console(inputs, pred_idx)
