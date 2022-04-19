import pandas as pd

TEMPL_STR = """

# Leaderboard  

## Test set  

{test_lb}

## Dev set

{dev_lb}

"""
def postprocess_leaderboard(df):
    df = df.round(2)
    for c in df.columns:
        max_idx = df[c].idxmax()
        max_score = df.at[max_idx, c]
        df.at[max_idx, c] = f"**{max_score}**"
    return df

def generate_readme():
    dev_lb = pd.read_csv("dev_leaderboard.csv", index_col=0)
    test_lb = pd.read_csv("test_leaderboard.csv", index_col=0)

    dev_lb = postprocess_leaderboard(dev_lb)
    test_lb = postprocess_leaderboard(test_lb)
    
    with open("README.MD", "w+") as f:
        f.write(TEMPL_STR.format(
            test_lb=test_lb.to_markdown(), 
            dev_lb=dev_lb.to_markdown(),
        ))


def main():
    generate_readme()

if __name__ == "__main__":
    main()