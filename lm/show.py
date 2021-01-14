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

    COLOR_CHAR = {
            "blue": '\033[32;1m',
            "red": '\033[31;1m',
            "close": '\033[0m'
        }

    def __init__(self, tokenizer: RobertaTokenizerFast, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        

    def on_evaluate(self, *args, model=None, eval_dataloader=None, **kwargs):
        with torch.no_grad():
            rand_example = randrange(eval_dataloader.batch_size)
            batch = next(iter(eval_dataloader))
            input_ids = batch['input_ids'][rand_example]
            attention_mask = batch['attention_mask'][rand_example]
            labels = batch['labels'][rand_example]
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            for k, v in inputs.items():
                inputs[k] = torch.tensor(v).unsqueeze(0)  # single example
                if torch.cuda.is_available():
                    inputs[k] = inputs[k].cuda()
            pred = model(**inputs)
            pred_idx = pred['logits'].argmax(-1)[0].cpu()
        pred_idx = [e.item() for e in pred_idx]
        colored = ""
        for i in range(len(input_ids)):
            input_id = input_ids[i]
            pred = pred_idx[i]
            masked = labels[i] != -100
            decoded = self.tokenizer.decode(pred) if masked else self.tokenizer.decode(input_id)
            if masked:
                color = "blue" if pred == input_id else "red"
                colored += f"{self.COLOR_CHAR[color]}{decoded}{self.COLOR_CHAR['close']}"
            elif attention_mask[i] == 1:
                colored += decoded
        print(f"\n\n{colored}\n\n")
