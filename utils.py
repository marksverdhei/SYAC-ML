import os
import json
import argparse
import re
from typing import Any, Dict, List, Union

from datasets import load_metric
import pandas as pd
import nltk
import toml
import torch
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint


class DocumentPreprocessor:
    """
    The preprocessor makes the data input on the preferred form of the model.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.sep_token = config.get("sep_token", None)
        self.sep_token_sub = config.get("sep_token_sub", " ")
        self.regex_args = config.get("regex_sub_args", None)
        self.cased = config.get("cased", True)
        self.qa_prompt = config.get("qa_prompt", None)

    def __call__(self, title: str, body: str) -> str:
        if self.qa_prompt is not None:
            if self.qa_prompt == 1:
                # 1: append question mark
                title = title_to_quesiton(title)
            elif self.qa_prompt == 2:
                # 2: truncate before interrogative
                title = title_to_quesiton2(title)
            else:
                raise NotImplementedError("Select qa prompt 1, 2 or None")

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


class SampleGenerationCallback(TrainerCallback):
    EXAMPLE_IDS = [
        # T: Apple exec says people who want this controversial feature should stop using iPhones
        # A: Sideloading apps
        "o72exh",
        # T: The 76ers' Newest Concessions Item Is Causing a Cultural Controversy on Social Media
        # A: Meat Pies being eaten with fork and knife instead of with hands
        "7en3iy",
        # T: The worst horror film of all time according to Stephen King
        # A: Blood Feast
        "q5vodo",
        # T: An 80-year-old mother didn't share her Wordle score. It may have saved her life.
        # A: A woman was taken hostage in her home in the morning, daughters became concerned when she didn't text them her daily Wordle score. Called police, guy was arrested, woman is ok
        "sq60po",
        # T: Mega Millions is up to $970 million—there’s one way to up the odds of winning, according to a Harvard statistics professor
        # A: Buy more tickets with different numbers. Literally the only way to improve odds in a game of pure chance.
        "l2rn92",
        # Diet with beer and pies
        "550o4v",
    ]

    "This callback produces title answers on a few examples"

    def __init__(self, eval_set_path, tokenizer, preprocess_fn) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.example_subset = pd.read_csv(eval_set_path, index_col=0).loc[
            self.EXAMPLE_IDS
        ]
        self.inputs = {
            post_id: tokenizer(
                preprocess_fn(row.title, row.body), return_tensors="pt", truncation=True
            )
            for post_id, row in self.example_subset.iterrows()
        }

    def on_evaluate(self, *_, **kwargs):
        model = kwargs["model"]
        for post_id, inputs in self.inputs.items():
            row = self.example_subset.loc[post_id]
            (output,) = model.generate(**inputs.to(model.device))
            prediction = self.tokenizer.decode(output.to("cpu"))

            print("=" * 20, "title", "=" * 20)
            print(row.title)
            print("=" * 20, "generated answer", "=" * 20)
            print(prediction)
            print("=" * 20, "target answer", "=" * 20)
            print(row.target)
            print("\n")


class EXTOracle:
    """
    Although we call this a baseline, it cannot implement the
    pipeline interface as it requires access to the labels.
    """

    name = "EXT-Oracle"
    rouge = load_metric("rouge")

    # def predict_on_df(self, df):
    #     return df.apply(self.predict_on_row, axis=1)
    def predict_on_row(self, row):
        body_sentences = nltk.tokenize.sent_tokenize(row.body)
        n_sents = len(body_sentences)
        targets = [row.target] * n_sents
        scores = self.rouge.compute(
            predictions=body_sentences,
            references=targets,
            rouge_types=["rouge1", "rouge2", "rougeL"],
            use_aggregator=False,
        )
        # selecting on sum of scores is the same as selecting on mean
        sum_scores = [sum(v[i].fmeasure for v in scores.values()) for i in range(n_sents)]
        argmax = max(range(n_sents), key=lambda x: sum_scores[x])
        return body_sentences[argmax]


def get_config_path_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="path to the config toml file of the script",
    )
    args = parser.parse_args()
    return args.config


def get_best_checkpoint(checkpoints_dir: str) -> str:
    "Reads the trainer state to find the best checkpoint"
    last_checkpoint_dir = get_last_checkpoint(checkpoints_dir)
    trainer_state_path = os.path.join(last_checkpoint_dir, "trainer_state.json")
    with open(trainer_state_path, "rb") as f:
        best_path = json.load(f)["best_model_checkpoint"]
    return best_path


def read_tsv(path):
    return pd.read_csv(path, index_col=0, sep="\t")


def read_config(fields: Union[str, List[str]] = None,) -> Dict[str, str]:
    """
    This function reads a specified config path from the command line
    if it is supplied
    """
    config_path = get_config_path_arg()
    conf = toml.load(config_path)

    if conf.get("dataset") is None:
        conf["dataset"] = {
            "train_path": "reddit-syac/train.csv",
            "val_path": "reddit-syac/dev.csv",
            "test_path": "reddit-syac/test.csv",
        }

    if fields is None:
        return conf
    elif isinstance(fields, list):
        subconf = {k: conf[k] for k in fields}
        return subconf
    else:
        return conf[fields]


def get_model_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def title_to_quesiton(title: str) -> str:
    """
    Tries to convert titles to questions

    Here's the reason why
    this is why
    the reason why x is verb y
    why is x verb why
    This is what you need to know

    what you need to know?

    here are the leaked relese dates
    what is the leaked release dates?
    """

    if title.endswith("?"):
        return title

    title = title + "?"
    return title


def title_to_quesiton2(title: str) -> str:
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

    if title.endswith("?"):
        return title

    match = re.search(title, "(" + "|".join(interrogatives) + ")")
    if match is None:
        return title
    else:
        s, _ = match.span()
        return title[s:] + "?"


# class ParaphraseProbabilityScorer:
#     """
#     This class is a model based scores for seq2seq tasks 
#     leveraging a paraphrase detection model
#     """

#     model_checkpoint = "coderpotter/adversarial-paraphrasing-detector"

#     def __init__(self):
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             self.model_checkpoint
#         )
#         self.model.eval()

#     @torch.no_grad()
#     def __call__(self, y_pred, y_true) -> Any:
#         inputs = self.tokenizer(y_pred, y_true, return_tensors="pt", padding=True)
#         outputs = self.model(**inputs)
#         scores = outputs.logits.softmax(dim=-1)
#         # Return probability for a paraphrase
#         return scores.T[1]
