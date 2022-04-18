import datasets
import pandas as pd
from utils import read_config
from datasets import load_metric


def evaluate_from_config(config):
    model_name = config["model"]["name"]
    dev_set_path = config["dataset"]["val_path"]
    test_set_path = config["dataset"]["test_path"]

    dev_set = pd.read_csv(dev_set_path, index_col=0)
    test_set = pd.read_csv(test_set_path, index_col=0)

    dev_predictions = pd.read_csv(f"evaluation/dev/{model_name}/predictions.csv", index_col=0)
    test_predictions = pd.read_csv(f"evaluation/test/{model_name}/predictions.csv", index_col=0)

    dev_set["prediction"] = dev_predictions
    test_set["prediction"] = test_predictions

    rouge = load_metric("rouge")
    bleu = load_metric("sacrebleu")
    meteor = load_metric("meteor")

    for split_name, df in [("dev", dev_set), ("test", test_set)]:
        scores = {}

        predictions = df["prediction"].array
        # Change shape to (500, 1)
        references = df["target"].array

        rouge_scores = rouge.compute(
            predictions=predictions,
            references=references,
        )

        scores.update({k: v.mid.fmeasure for k, v in rouge_scores.items()})

        bleu_score = bleu.compute(
            predictions=predictions,
            references=references[:, None],
        )

        scores["bleu"] = bleu_score["score"]

        meteor_score = meteor.compute(
            predictions=predictions,
            references=references,
        )

        scores.update(meteor_score)

        scores_df = pd.DataFrame(scores, index=["example-model"])
        scores_df.index.name = "Model"
        scores_df.to_csv(f"evaluation/{split_name}/{model_name}/scores.csv")


def main() -> None:
    config = read_config()
    evaluate_from_config(config)


if __name__ == "__main__":
    main()
