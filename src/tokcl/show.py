from transformers import TrainerCallback, RobertaTokenizerFast
from random import randrange
import torch

# uses spcial color characters for the console output
# https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html#colors
# for code in {1..256}; do printf "\e[38;5;${code}m"$code"\e[0m";echo; done
# for i = 1, 32 do COLORS[i] = "\27[38;5;"..(8*i-7).."m" end
# printf "\e[30;1mTesting color\e[0m"
# for i in range(25,50): print(f"\033[{i};1mTesting color {i}\033[0m")
# for i in range(256): print(f"\033[38;5;{i}mBlahblah color={i}\033[0mAnd normal")


class ShowExample(TrainerCallback):
    """Visualizes on the console the result of a prediction with the current state of the model.
    It uses a randomly picked input example and decodes the input with the provided tokenizer.
    Words are colored depending on the predicted class. Note that B- and I- IOB labels will have different colors.

    Args:

        tokenizer (RobertaTokenizer): the tokenizer used to generate the dataset.
    """

    UNDERSCORE = "\033[4m"
    BOLD = "\033[1m"
    CLOSE = "\033[0m"
    COLOR = "\033[38;5;{color_idx}m"

    def __init__(self, tokenizer: RobertaTokenizerFast, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def on_evaluate(self, *args, model=None, eval_dataloader=None, **kwargs):
        batch = next(iter(eval_dataloader))
        rand_example = randrange(batch['input_ids'].size(0))
        input_ids = batch['input_ids'][rand_example]
        attention_mask = batch['attention_mask'][rand_example]
        labels = batch['labels'][rand_example]
        inputs = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
        with torch.no_grad():
            for k, v in inputs.items():
                inputs[k] = v.clone().unsqueeze(0)  # single example
                if torch.cuda.is_available():
                    inputs[k] = inputs[k].cuda()
            pred = model(**inputs)
            labels_idx = pred['logits'].argmax(-1)[0].cpu()
            input_ids = inputs['input_ids'][0].cpu()
        labels_idx = [e.item() for e in labels_idx]
        input_ids = [e.item() for e in input_ids]
        colored = ""
        for i in range(len(input_ids)):
            input_id = input_ids[i]
            label_idx = labels_idx[i]
            true_label = labels[i].item()
            if input_id != self.tokenizer.pad_token_id:  # don't display padding
                decoded = self.tokenizer.decode(input_id)
                # indicate the true label with underline
                underscore = self.UNDERSCORE if true_label > 0 else ''
                if label_idx > 0:
                    colored += f"{self.BOLD}{underscore}{self.COLOR.format(color_idx=label_idx)}{decoded}{self.CLOSE}"
                else:
                    colored += f"{underscore}{decoded}{self.CLOSE}"
        print(f"\n\n{colored}\n\n")
