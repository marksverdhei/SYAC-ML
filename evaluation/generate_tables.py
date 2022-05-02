import os
from argparse import ArgumentParser
import pandas as pd

TEMPL_STR = """

# Leaderboard  

## Dev set

{dev_lb}
"""

TEMPL_STR_ALL = (
    TEMPL_STR
    + """

## Test set  

{test_lb}
"""
)


def assemble_subdirs(leaderboard, split, mode):
    dirs = os.listdir(split)
    if mode == 1:
        dirs = filter(lambda s: "Baseline" in s, dirs)
    elif mode == 2:
        dirs = filter(lambda s: "zero" in s, dirs)

    for dir in dirs:
        for suffix in "", "-1024", "-2048":
            scores_path = f"{split}/{dir}/scores{suffix}.csv"
            if os.path.exists(scores_path):
                scores = pd.read_csv(scores_path, index_col=0)
                scores.index = [i + suffix for i in scores.index]
                leaderboard = pd.concat((leaderboard, scores))
        # else:
        #     print("WANRING: did not find", scores_path, ", skipping")

    return leaderboard


def postprocess_leaderboard(df, format="md"):
    df = df.round(2)
    for c in df.columns:
        max_score = df[c].max()
        max_idxs = df[df[c] == max_score].index
        df[c] = df[c].astype("string")
        for max_idx in max_idxs:
            if format == "md":
                df.at[max_idx, c] = f"**{max_score}**"
            elif format == "latex":
                df.at[max_idx, c] = r"\textbf{" + str(max_score) + "}"
            else:
                raise NotImplementedError("only supports md and latex")
    return df


def create_leaderboard_df(split, mode):
    leaderboard = pd.DataFrame()

    leaderboard = assemble_subdirs(leaderboard, split, mode)
    leaderboard = leaderboard[["rouge1", "rouge2", "rougeL", "bleu", "meteor"]]
    leaderboard[["rouge1", "rouge2", "rougeL", "meteor"]] *= 100
    return leaderboard


def generate_table(*, format="md", aggregate_baselines=False, test=False, mode=0):
    dev_lb = create_leaderboard_df(
        "dev",
        mode=mode,
    )
    dev_lb = postprocess_leaderboard(dev_lb, format=format)

    if test:
        test_lb = create_leaderboard_df(
            "test", 
            mode=mode,
        )
        test_lb = postprocess_leaderboard(test_lb, format=format)

    if format == "md":
        with open("README.md", "w+") as f:
            if test:
                f.write(
                    TEMPL_STR_ALL.format(
                        dev_lb=dev_lb.to_markdown(), 
                        test_lb=test_lb.to_markdown(),
                    )
                )
            else:
                f.write(TEMPL_STR.format(dev_lb=dev_lb.to_markdown(),))
    else:
        dev_lb.to_latex("dev_leaderboard.tex", escape=False)
        if test:
            test_lb.to_latex("test_leaderboard.tex", escape=False)


def main():
    ap = ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--aggregate-baselines", action="store_true")
    ap.add_argument(
        "--mode",
        type=int,
        default=0,
        help="0: all, 1: baselines only, 2: zero shot only, 3: fine tune only",
    )
    args = ap.parse_args()
    print(args.mode)

    format = "latex" if args.latex else "md"
    generate_table(
        format=format,
        test=args.test,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
