import os
import pandas as pd
from utils import read_config
from tqdm import tqdm
from pipelines import TitleAnsweringPipeline

tqdm.pandas()


def predict_from_config(config):
    model_name = config["model"]["name"]
    dev_set_path = config["dataset"]["val_path"]
    test_set_path = config["dataset"]["test_path"]

    pipeline = TitleAnsweringPipeline.from_config(config)
    pipeline.tokenizer.model_max_length = 512


    dev_predictions, test_predictions = predict_on_datasets(dev_set_path, test_set_path, pipeline)

    if not os.path.exists(f"evaluation/dev/{model_name}"):
        os.mkdir(f"evaluation/dev/{model_name}")

    if not os.path.exists(f"evaluation/test/{model_name}"):
        os.mkdir(f"evaluation/test/{model_name}")

    dev_predictions.to_csv(f"evaluation/dev/{model_name}/predictions.csv")
    test_predictions.to_csv(f"evaluation/test/{model_name}/predictions.csv")


def predict_on_datasets(dev_set_path, test_set_path, pipeline):
    dev_set = pd.read_csv(dev_set_path, index_col=0)
    test_set = pd.read_csv(test_set_path, index_col=0)

    def predict_on_row(row):
        return pipeline(row.title, row.body)

    dev_predictions = dev_set.progress_apply(predict_on_row, axis=1).rename(pipeline.name)
    test_predictions = test_set.progress_apply(predict_on_row, axis=1).rename(pipeline.name)

    return dev_predictions, test_predictions


def main() -> None:
    config = read_config()
    predict_from_config(config)


if __name__ == "__main__":
    main()
