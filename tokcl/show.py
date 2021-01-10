from transformers import TrainerCallback, RobertaTokenizerFast
from random import randrange
import torch


class ShowExample(TrainerCallback):

    def __init__(self, tokenizer: RobertaTokenizerFast, label_list: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.label_list = label_list

    def on_evaluate(self, *args, model=None, eval_dataloader=None, **kwargs):
        N = len(eval_dataloader.dataset)
        idx = randrange(N)
        with torch.no_grad():
            inputs = eval_dataloader.dataset[idx]
            for k, v in inputs.items():
                inputs[k] = torch.tensor(v).unsqueeze(0)
                if torch.cuda.is_available():
                    inputs[k] = inputs[k].cuda()
            # inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
            pred = model(**inputs)
            labels_idx = pred['logits'].argmax(-1)[0].cpu()
            input_ids = inputs['input_ids'][0].cpu()
        labels_idx = [e.item() for e in labels_idx]
        input_ids = [e.item() for e in input_ids]
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        # print(f"\n\nExample: {self.tokenizer.decode(input_ids)}")
        # for i in range(len(input_ids)):
        #     print(f"{i}\t{tokens[i]}\t{self.label_list[labels_idx[i]]}")
        colored = ""
        for input_id, label_idx in zip(input_ids, labels_idx):
            decoded = self.tokenizer.decode(input_id)
            colored += f"{'\033[38;5;'+str(label_idx)+'m' if label_idx != 0 else ''}{decoded}{'\033[0m' if label_idx !=0 else ''}"
        print(f"\n{colored}\n")


# decode() -> the whole string
# word_to_tokens: to get the token idx and get the label
# word_to_chars -> (start, end) text[start:end]
# for code in {1..256}; do printf "\e[38;5;${code}m"$code"\e[0m";echo; done
#for i = 1, 32 do COLORS[i] = "\27[38;5;"..(8*i-7).."m" end
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