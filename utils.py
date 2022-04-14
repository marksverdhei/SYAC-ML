import argparse
import re
from typing import Any, Dict, List, Union

import pandas as pd
import toml
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


CONFIG_PATH = "config.toml"


class DocumentPreprocessor:
    """
    The preprocessor makes the data input on the preferred form of the model.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.sep_token = config.get("sep_token", None)
        self.sep_token_sub = config.get("sep_token_sub", "")
        self.regex_args = config.get("regex_sub_args", None)
        self.cased = config.get("cased", True) 

    def __call__(self, title: str, body: str) -> str:
        if self.sep_token is not None:
            title = title.replace(self.sep_token, self.sep_token_sub)
            body = body.replace(self.sep_token, self.sep_token_sub)
            sep_token = self.sep_token
        else:
            sep_token = " "

        input_str = f"{title} {sep_token} {body}"

        if not self.cased:
            input_str = input_str.lower()

        if self.regex_args is not None:
            regex_pattern, regex_repl = self.regex_args
            input_str = re.sub(regex_pattern, regex_repl, input_str)

        return input_str


class ParaphraseProbabilityScorer:
    """
    This class is a model based scores for seq2seq tasks 
    leveraging a paraphrase detection model
    """
    model_checkpoint = "coderpotter/adversarial-paraphrasing-detector"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, y_pred, y_true) -> Any:
        inputs = self.tokenizer(y_pred, y_true, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        # Return probability for a paraphrase
        return scores.T[1]


def get_config_path_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        default=CONFIG_PATH, 
        help="path to the config toml file of the script",
    )
    args = parser.parse_args()
    return args.config


def read_tsv(path):
    return pd.read_csv(path, index_col=0, sep="\t")


def read_config(
    fields: Union[str, List[str]],
) -> Dict[str, str]:
    """
    This function reads a specified config path from the command line
    if it is supplied
    """
    config_path = get_config_path_arg()
    conf = toml.load(config_path)

    if isinstance(fields, list):
        subconf = {k: conf[k] for k in fields}
        return subconf
    else:
        return conf[fields]


def get_model_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def title_to_quesiton(title: str) -> str:
    raise NotImplementedError("TODO")
    interrogatives = [
        "which", 
        "what",
        "whose",
        "who",
        "what",
        "where",
        "when",
        "how",
        "why",
    ]

    implicit_question_patterns = [
        "this"
    ]

    # TODO

    if title.endswith("?"):
        return title

    title = title + "?"

    title_lower = title.lower()
    if title_lower.startswith("why"):
        title = title.split(" ")
        swp = title[1]
        title[1] = title[2]
        title[2] = swp
        return " ".join(title)

    if "reason why" in title_lower:
        title = title.split(" ")
        slice_index = title_lower.split(" ").index("reason")
        return " ".join(title[slice_index:])

    title_interrogatives = set(filter(lambda x: x in title_lower, interrogatives))
    if title_interrogatives:
        print(title_interrogatives)
    else:
        title = title.replace("this", "what")

    return title
