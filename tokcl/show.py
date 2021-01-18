from transformers import TrainerCallback, RobertaTokenizerFast
from random import randrange
from typing import List
import torch

# uses spcial color characters for the console output
# for code in {1..256}; do printf "\e[38;5;${code}m"$code"\e[0m";echo; done
# for i = 1, 32 do COLORS[i] = "\27[38;5;"..(8*i-7).."m" end
# printf "\e[30;1mTesting color\e[0m"
# for i in range(25,50): print(f"\033[{i};1mTesting color {i}\033[0m")
# for i in range(256): print(f"\033[38;5;{i}mBlahblah color={i}\033[0mAnd normal")

# "\033[48;1m", #grey
# "\033[34;1m", #blue first since often assayed
# "\033[31;1m", #red
# "\033[33;1m", #yellow
# "\033[32;1m", #green
# "\033[35;1m", #pink
# "\033[36;1m", #turquoise
# "\033[41;37;1m", #red back
# "\033[42;37;1m", #green back
# "\033[43;37;1m", #yellow back
# "\033[44;37;1m", #blue back
# "\033[45;37;1m" #turquoise back
# "\033[0m", # close


class ShowExample(TrainerCallback):
    """Visualizes on the console the result of a prediction with the current state of the model.
    It uses a randomly picked input example and decodes the input with the provided tokenizer.
    Words are colored depending on the predicted class. Note that B- and I- IOB labels will have different colors.

    Args:

        tokenizer (RobertaTokenizer): the tokenizer used to generate the dataset.
    """

    UNDERSCORE = "\033[4m"
    CLOSE = "\033[0m"
    COLOR = "\033[38;5;{color_idx}m"

    def __init__(self, tokenizer: RobertaTokenizerFast, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def on_evaluate(self, *args, model=None, eval_dataloader=None, **kwargs):
        N = len(eval_dataloader.dataset)
        idx = randrange(N)
        with torch.no_grad():
            inputs = eval_dataloader.dataset[idx]
            for k, v in inputs.items():
                inputs[k] = torch.tensor(v).unsqueeze(0)
                if torch.cuda.is_available():
                    inputs[k] = inputs[k].cuda()
            pred = model(**inputs)
            labels_idx = pred['logits'].argmax(-1)[0].cpu()
            input_ids = inputs['input_ids'][0].cpu()
        labels_idx = [e.item() for e in labels_idx]
        input_ids = [e.item() for e in input_ids]
        colored = ""
        for input_id, label_idx in zip(input_ids, labels_idx):
            decoded = self.tokenizer.decode(input_id)
            if label_idx > 0:
                colored += f"{self.UNDERSCORE}{self.COLOR.format(color_idx=label_idx)}{decoded}{self.CLOSE}"
            else:
                colored += decoded
        print(f"\n\n{colored}\n\n")
