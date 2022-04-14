import pandas as pd
from utils import read_config
from datasets import load_metric
from tqdm import tqdm
from baselines import BASELINES

EVALUATION_REPORT_TEMPLATE = """
# Evaluation of {model_name}

{scores}

"""

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
MEASURES = ["precision", "recall", "fmeasure"]

rouge = load_metric("rouge")

def compute_rouge(y_true, y_pred):
    output = rouge.compute(
        predictions=[y_pred], 
        references=[y_true])
    
    return {
        rouge_type: output[rouge_type].mid.fmeasure 
        for rouge_type in ROUGE_TYPES
    }


def evaluate_on_dataset(eval_set, model, metrics, progress=True):
    scores = []

    row_iter = eval_set.iterrows()
    if progress:
        row_iter = tqdm(row_iter, total=len(eval_set))

    for _, row in row_iter:
        answer = model(
            title=row.title,
            body=row.body, 
        )
        answer = answer.lower()
        target = row.target.lower()

        scores.append({name: f(answer, target) for name, f in metrics.items()})

    return scores

def run_evaluation(
    eval_set,
    pipelines,
):
    metric_functions = {
        "rouge1_f1": compute_rouge,
    }

    results_per_model = {
        pipeline_name: evaluate_on_dataset(eval_set, pipeline, metric_functions)
        for pipeline_name, pipeline in pipelines.items()
    }

    return results_per_model  

def write_metric_report(report_filename, tables) -> None:
    md_tables = {k: v.to_markdown() for k, v in tables.items()}

    with open(report_filename, "w+") as f:
        f.write(EVALUATION_REPORT_TEMPLATE.format(**md_tables))


def main() -> None:
    config = read_config(["evaluate", "datapaths"])
    dataset_type = config["evaluate"]["dataset_type"]
    eval_set_path = config["datapaths"][f"{dataset_type}_path"]
    model_type = config["evaluate"]["model"]
    model_path = config["evaluate"]["model_path"]
    report_path = config["evaluate"]["report_path"]

    eval_sets = [
        # pd.read_csv(eval_set_path, index_col=0, sep="\t")
        pd.read_csv("../data/test_set.tsv", index_col=0, sep="\t")
    ]
    
    # model = TitleAnsweringUnifiedQAPipeline(model_path=model_path, tokenizer_path="allenai/unifiedqa-t5-base")
    report = ""
    # TODO: swap out baselines
    for eval_set in eval_sets:
        scores = run_evaluation(
            eval_set=eval_set,
            pipelines=BASELINES,
        )

        for model_name in BASELINES:
            baseline_scores = [i["rouge1_f1"] for i in scores[model_name]]

            # FIXME
            scores_df = pd.DataFrame(baseline_scores).mean(axis=0)
            scores_md = scores_df.to_markdown()

            report += EVALUATION_REPORT_TEMPLATE.format(
                model_name=model_name,
                scores=scores_md,
            )

    with open("baseline_scores_test_set.md", "w+") as f:
        f.write(report)


if __name__ == "__main__":
    main()
