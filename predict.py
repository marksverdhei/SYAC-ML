import os
import pandas as pd
from utils import read_config, EXTOracle
from tqdm import tqdm
from pipelines import TitleAnsweringPipeline
from baselines import BASELINES


tqdm.pandas()


def predict_from_config(config):
    model_name = config["model"]["name"]
    dev_set_path = config["dataset"]["val_path"]
    test_set_path = config["dataset"]["test_path"]

    pipeline = TitleAnsweringPipeline.from_config(config)
    pipeline.tokenizer.model_max_length = 512

    dev_predictions = predict_on_df(
        dev_set_path, pipeline
    )

    if not os.path.exists(f"evaluation/dev/{model_name}"):
        os.mkdir(f"evaluation/dev/{model_name}")

    dev_predictions.to_csv(f"evaluation/dev/{model_name}/predictions.csv")

    test_predictions = predict_on_df(
        test_set_path, pipeline
    )

    if not os.path.exists(f"evaluation/test/{model_name}"):
        os.mkdir(f"evaluation/test/{model_name}")

    test_predictions.to_csv(f"evaluation/test/{model_name}/predictions.csv")


def predict_on_df(eval_set_path, pipeline):
    eval_set = pd.read_csv(eval_set_path, index_col=0)

    def predict_on_row(row):
        return pipeline(row.title, row.body)

    eval_predictions = eval_set.progress_apply(predict_on_row, axis=1).rename(
        pipeline.name
    )

    return eval_predictions


def predict_baselines():
    oracle = EXTOracle()

    dev_set_path = "reddit-syac/dev.csv"
    test_set_path = "reddit-syac/test.csv"

    paths_dict = {
        "dev": dev_set_path,
        "test": test_set_path,
    }

    for split in "dev", "test":
        eval_set_path = paths_dict[split]

        for name, pipeline in BASELINES.items():
            preds_path = f"evaluation/{split}/{name}/"
            if not os.path.exists(preds_path):
                os.mkdir(preds_path)

            dev_pred = predict_on_df(eval_set_path, pipeline)
            dev_pred.to_csv(preds_path + "predictions.csv")


        preds_path = f"evaluation/{split}/EXTOracle/"
        if not os.path.exists(preds_path):
            os.mkdir(preds_path)

        eval_set = pd.read_csv(eval_set_path, index_col=0)
        test_pred = eval_set.progress_apply(oracle.predict_on_row, axis=1)
        test_pred.to_csv(preds_path + "predictions.csv")

def main() -> None:
    predict_on_baselines = False
    if predict_on_baselines:
        predict_baselines()
    else:
        config = read_config()
        predict_from_config(config)



if __name__ == "__main__":
    main()
