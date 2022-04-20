import os
from argparse import ArgumentParser
import pandas as pd

TEMPL_STR = """

# Leaderboard  

## Dev set

{dev_lb}
"""

TEMPL_STR_ALL = TEMPL_STR + """

## Test set  

{test_lb}
"""

def assemble_subdirs(leaderboard, split):
    for dir in os.listdir(split):
        scores_path = f"{split}/{dir}/scores.csv"
        if os.path.exists(scores_path):
            scores = pd.read_csv(scores_path, index_col=0)
            leaderboard = pd.concat((leaderboard, scores))
        else:
            print("WANRING: did not find", scores_path, ", skipping")

    return leaderboard


def postprocess_leaderboard(df, format="md"):
    df = df.round(2)
    for c in df.columns:
        max_idx = df[c].idxmax()
        max_score = df.at[max_idx, c]
        if format == "md":
            df.at[max_idx, c] = f"**{max_score}**"
        elif format == "latex":
            df.at[max_idx, c] = r"\textbf{" + str(max_score) + "}"
        else:
            raise NotImplementedError("only supports md and latex")
    return df


def create_leaderboard_df(split, aggregate_baselines=False):
    leaderboard = pd.read_csv(f"{split}_leaderboard.csv", index_col=0)
    if aggregate_baselines:
        leaderboard = leaderboard.max().to_frame("Best baseline").T

    leaderboard = assemble_subdirs(leaderboard, split)
    
    return leaderboard


def generate_table(*, format="md", aggregate_baselines=False, test=False):
    dev_lb = create_leaderboard_df("dev", aggregate_baselines=aggregate_baselines)
    dev_lb = postprocess_leaderboard(dev_lb, format=format)


    if test:
        test_lb = create_leaderboard_df("test", aggregate_baselines=aggregate_baselines)
        test_lb = postprocess_leaderboard(test_lb, format=format)

    if format == "md":
        with open("README.md", "w+") as f:
            if test:
                f.write(TEMPL_STR_ALL.format(
                    dev_lb=dev_lb.to_markdown(),
                    test_lb=test_lb.to_markdown(),
                ))
            else:        
                f.write(TEMPL_STR.format(
                    dev_lb=dev_lb.to_markdown(),
                ))
    else:
        dev_lb.to_latex("dev_leaderboard.tex", escape=False)
        if test:
            test_lb.to_latex("test_leaderboard.tex", escape=False)


def main():
    ap = ArgumentParser()
    ap.add_argument("--test", action='store_true')
    ap.add_argument("--latex", action='store_true')
    args = ap.parse_args()

    format = "latex" if args.latex else "md"
    generate_table(format=format, test=args.test)


if __name__ == "__main__":
    main()