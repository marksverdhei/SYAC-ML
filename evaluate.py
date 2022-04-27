import os
import pandas as pd
from utils import EXTOracle, read_config
from datasets import load_metric
from baselines import BASELINES
from tqdm import tqdm
from predict import predict_on_df
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

    dev_predictions = pd.read_csv(
        f"evaluation/dev/{model_name}/predictions.csv", index_col=0
    )
    test_predictions = pd.read_csv(
        f"evaluation/test/{model_name}/predictions.csv", index_col=0
    )

    dev_scores, test_scores = evaluate_predictions(
        (dev_set, test_set), (dev_predictions, test_predictions), model_name
    )
    dev_scores.to_csv(f"evaluation/dev/{model_name}/scores.csv")
    test_scores.to_csv(f"evaluation/test/{model_name}/scores.csv")


def evaluate_predictions(datasets, predictions, model_name="Unnamed model"):
    for df, preds in zip(datasets, predictions):
        df["prediction"] = preds
        yield compute_metrics(df, model_name)


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


def evaluate_baselines():
    oracle = EXTOracle()

    dev_set_path = "reddit-syac/dev.csv"
    test_set_path = "reddit-syac/test.csv"

    result_dev_df = pd.DataFrame()
    result_test_df = pd.DataFrame()

    dev_set = pd.read_csv(dev_set_path, index_col=0)
    test_set = pd.read_csv(test_set_path, index_col=0)

    for name, pipeline in BASELINES.items():
        dev_pred = predict_on_df(dev_set_path, pipeline)
        test_pred = predict_on_df(test_set_path, pipeline)
        dev_set["prediction"] = dev_pred
        test_set["prediction"] = test_pred
        dev_metrics = compute_metrics(dev_set, name)
        test_metrics = compute_metrics(test_set, name)
        result_dev_df = pd.concat((result_dev_df, dev_metrics))
        result_test_df = pd.concat((result_test_df, test_metrics))

    dev_set["prediction"] = dev_set.progress_apply(oracle.predict_on_row, axis=1)
    test_set["prediction"] = test_set.progress_apply(oracle.predict_on_row, axis=1)

    dev_metrics = compute_metrics(dev_set, "EXT-Oracle")
    test_metrics = compute_metrics(test_set, "EXT-Oracle")
    result_dev_df = pd.concat((result_dev_df, dev_metrics))
    result_test_df = pd.concat((result_test_df, test_metrics))

    result_dev_df.to_csv("evaluation/dev_leaderboard.csv")
    result_test_df.to_csv("evaluation/test_leaderboard.csv")


def main() -> None:
    set_seed(42)
    recursive = True
    eval_baselines = False

    if eval_baselines:
        evaluate_baselines()

    elif recursive:
        splits = ["dev"]
        splits = ["dev", "test"]

        for s in splits:
            eval_set_path = f"reddit-syac/{s}.csv"
            eval_set = eval_set = pd.read_csv(eval_set_path, index_col=0)
            pred_dir = f"evaluation/{s}/"

            for model_name in os.listdir(pred_dir):
                pred_path = pred_dir + model_name + "/predictions.csv"
                scores_path = pred_dir + model_name + "/scores.csv"
                if not os.path.exists(pred_path):
                    print("WARNNING: didnt find predictions for ", model_name)

                if not os.path.exists(scores_path):
                    predictions = pd.read_csv(pred_path, index_col=0)
                    # FIXME: ugly hack for unsolved bug
                    predictions = predictions[:500]
                    (scores,) = evaluate_predictions(
                        [eval_set], [predictions], model_name
                    )
                    scores.to_csv(scores_path)

    else:
        config = read_config()
        evaluate_from_config(config)


if __name__ == "__main__":
    main()
