import pandas as pd
from utils import read_config
from datasets import load_metric
from baselines import BASELINES
from tqdm import tqdm
from predict import predict_on_datasets

tqdm.pandas()

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

    dev_scores, test_scores = compute_metrics(dev_set, test_set, model_name)
    dev_scores.to_csv(f"evaluation/dev/{model_name}/scores.csv")
    test_scores.to_csv(f"evaluation/test/{model_name}/scores.csv")


def compute_metrics(dev_set, test_set, model_name):
    rouge = load_metric("rouge")
    bleu = load_metric("sacrebleu")
    meteor = load_metric("meteor")
    scores_dfs = []

    for df in (dev_set, test_set):
        scores = {}
        predictions = df["prediction"].array
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

        scores_df = pd.DataFrame(scores, index=[model_name])
        scores_df.index.name = "Model"
        scores_dfs.append(scores_df)

    return scores_dfs


def evaluate_baselines():
    dev_set_path = "reddit-syac/dev.csv"
    test_set_path = "reddit-syac/test.csv"
    
    result_dev_df = pd.DataFrame()
    result_test_df = pd.DataFrame()
    
    for name, pipeline in BASELINES.items():
        dev_set = pd.read_csv(dev_set_path, index_col=0)
        test_set = pd.read_csv(test_set_path, index_col=0)
        dev_pred, test_pred = predict_on_datasets(dev_set_path, test_set_path, pipeline)
        dev_set["prediction"] = dev_pred
        test_set["prediction"] = test_pred
        dev_metrics, test_metrics = compute_metrics(dev_set, test_set, name)
        result_dev_df = pd.concat((result_dev_df, dev_metrics))
        result_test_df = pd.concat((result_test_df, test_metrics))

    result_dev_df.to_csv("evaluation/dev_leaderboard.csv")
    result_test_df.to_csv("evaluation/test_leaderboard.csv")



def main() -> None:
    eval_baselines = False

    if eval_baselines:
        evaluate_baselines()
    else:
        config = read_config()
        evaluate_from_config(config)


if __name__ == "__main__":
    main()
