from sklearn.metrics import adjusted_rand_score
from transformers import TrainerCallback, RobertaTokenizerFast
from random import randrange
import math
import torch

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
    """

    COLOR_CHAR = {
            "blue": '\033[32;1m',
            "red": '\033[31;1m',
            "close": '\033[0m'
        }

    def __init__(self, tokenizer: RobertaTokenizerFast, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def on_evaluate(self, *args, model=None, eval_dataloader=None, **kwargs):
        """Method called when evaluating the model. Only the neede args are unpacked.

        Args:

            model: the current model being trained.
            eval_dataloader (torch.utils.data.DataLoader): the DataLoader used to produce the evaluation examples
        """
        with torch.no_grad():
            batch = next(iter(eval_dataloader))
            rand_example = randrange(batch['input_ids'].size(0))
            input_ids = batch['input_ids'][rand_example]
            attention_mask = batch['attention_mask'][rand_example]
            labels = batch['labels'][rand_example]
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            for k, v in inputs.items():
                inputs[k] = v.clone().unsqueeze(0)  # single example
                if torch.cuda.is_available():
                    inputs[k] = inputs[k].cuda()
            pred = model(**inputs)
            pred_idx = pred['logits'].argmax(-1)[0].cpu()
        pred_idx = [e.item() for e in pred_idx]
        input_ids = [e.item() for e in input_ids]
        labels = [e.item() for e in labels]
        colored = ""
        for i in range(len(input_ids)):
            input_id = input_ids[i]
            pred_id = pred_idx[i]
            label = labels[i]
            is_prediction = label != -100
            if is_prediction:
                colored += self._correct_incorrect(pred_id, label)
            elif attention_mask[i] == 1:
                colored += self.tokenizer.decode(input_id)
        print(f"\n\n{colored}\n\n")
        print("raw prediction:")
        print(self.tokenizer.decode(pred_idx))
        # if 'adjascency' in pred:
        #     adjascency = pred['adjascency']
        #     mu = adjascency.mean()
        #     sd = adjascency.std()
        #     adjascency = adjascency > (mu + sd)
        #     _, N = adjascency.size()
        #     N = int(math.sqrt(N))
        #     adjascency = adjascency[0].view(N, N)
        #     adjascency = adjascency.int()
        #     for i in range(N):
        #         print(''.join([str(e.item()) for e in adjascency[i]]))

    def _correct_incorrect(self, pred_id, label):
        decoded_pred = self.tokenizer.decode(pred_id)
        decoded_label = self.tokenizer.decode(label)
        correct = (pred_id == label)
        color = "blue" if correct else "red"
        insert = decoded_pred if correct else f"{decoded_pred}[{decoded_label.strip()}]"
        colored = f"{self.COLOR_CHAR[color]}{insert}{self.COLOR_CHAR['close']}"
        return colored
