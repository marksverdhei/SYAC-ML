import os
import pandas as pd
from utils import read_config
from datasets import load_metric
from tqdm import tqdm
from transformers import set_seed

rouge = load_metric("rouge")
bleu = load_metric("sacrebleu")
meteor = load_metric("meteor")

tqdm.pandas()


def evaluate_from_config(config):
    model_name = config["model"]["name"]
    dev_set_path = config["dataset"]["val_path"]
    test_set_path = config["dataset"]["test_path"]

    dev_set = pd.read_csv(dev_set_path, index_col=0)
    test_set = pd.read_csv(test_set_path, index_col=0)

    # hardcoded, ok for now
    dataset_dict = {
        "dev": dev_set,
        "test": test_set,
    }

    for split in "dev", "test":
        eval_set = dataset_dict[split]
        model_dir = f"evaluation/{split}/{model_name}"
        eval_model_dir(eval_set, model_dir)


def eval_model_dir(eval_set, model_dir):
    for suffix in "", "-1024", "-2048":
        model_name = model_dir.split("/")[-1]
        pred_path = f"{model_dir}/predictions{suffix}.csv"
        if os.path.exists(pred_path):
            predictions = pd.read_csv(
                f"{model_dir}/predictions{suffix}.csv", index_col=0
            )
            eval_set["prediction"] = predictions
            scores = compute_metrics(eval_set, model_name)
            scores.to_csv(f"{model_dir}/scores{suffix}.csv")


def compute_metrics(df, model_name):
    scores = {}
    predictions = df["prediction"].array
    references = df["target"].array

    rouge_scores = rouge.compute(predictions=predictions, references=references,)
    scores.update({k: v.mid.fmeasure for k, v in rouge_scores.items()})

    bleu_score = bleu.compute(predictions=predictions, references=references[:, None],)
    scores["bleu"] = bleu_score["score"]

    meteor_score = meteor.compute(predictions=predictions, references=references,)
    scores.update(meteor_score)

    scores_df = pd.DataFrame(scores, index=[model_name])
    scores_df.index.name = "Model"
    print(scores_df)
    return scores_df


def main() -> None:
    set_seed(42)
    recursive = True

    if recursive:
        splits = ["dev"]
        splits = ["dev", "test"]

        for s in splits:
            eval_set_path = f"reddit-syac/{s}.csv"
            eval_set = eval_set = pd.read_csv(eval_set_path, index_col=0)
            pred_dir = f"evaluation/{s}/"

            for model_name in os.listdir(pred_dir):
                model_dir = pred_dir + model_name
                eval_model_dir(eval_set, model_dir)
                    # predictions = pd.read_csv(pred_path, index_col=0)
                    # # FIXME: ugly hack for unresolved bug
                    # if len(predictions) > 500:
                    #     print("Warning: predictions longer than 500, slicing")
                    #     predictions = predictions[:500]
    else:
        config = read_config()
        evaluate_from_config(config)


if __name__ == "__main__":
    main()
