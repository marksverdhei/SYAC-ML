import os

import pandas as pd

TEMPL_STR = """

# Leaderboard  

## Test set  

{test_lb}

## Dev set

{dev_lb}

"""
def assemble_subdirs(dev_lb, test_lb):
    for dir in os.listdir("dev"):
        dev_scores = pd.read_csv(f"dev/{dir}/scores.csv", index_col=0)
        dev_lb = pd.concat((dev_lb, dev_scores))

    for dir in os.listdir("test"):
        test_scores = pd.read_csv(f"test/{dir}/scores.csv", index_col=0)
        test_lb = pd.concat((test_lb, test_scores))

    return dev_lb, test_lb

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


def generate_table(format="md"):
    dev_lb = pd.read_csv("dev_leaderboard.csv", index_col=0)
    test_lb = pd.read_csv("test_leaderboard.csv", index_col=0)

    dev_lb, test_lb = assemble_subdirs(dev_lb, test_lb)

    dev_lb = postprocess_leaderboard(dev_lb, format=format)
    test_lb = postprocess_leaderboard(test_lb, format=format)
    
    if format == "md":
        with open("README.md", "w+") as f:
            f.write(TEMPL_STR.format(
                test_lb=test_lb.to_markdown(), 
                dev_lb=dev_lb.to_markdown(),
            ))
    else:
        dev_lb.to_latex("dev_leaderboard.tex", escape=False)
        test_lb.to_latex("test_leaderboard.tex", escape=False)


def main():
    generate_table()
    # generate_table(format="latex")


if __name__ == "__main__":
    main()