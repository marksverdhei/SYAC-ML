

# Leaderboard  

## Test set  

| Model                    | rouge1   | rouge2   | rougeL   | rougeLsum   | bleu     | meteor   |
|:-------------------------|:---------|:---------|:---------|:------------|:---------|:---------|
| TitleBaseline            | 0.07     | 0.01     | 0.06     | 0.06        | 0.44     | 0.06     |
| MostCommonAnswerBaseline | 0.03     | 0.0      | 0.03     | 0.03        | 0.0      | 0.01     |
| CosineSimilarityBaseline | **0.1**  | **0.03** | **0.09** | **0.09**    | **1.95** | **0.13** |
| EditDistanceBaseline     | 0.08     | 0.01     | 0.07     | 0.07        | 1.03     | 0.08     |

## Dev set

| Model                    | rouge1   | rouge2   | rougeL   | rougeLsum   | bleu     | meteor   |
|:-------------------------|:---------|:---------|:---------|:------------|:---------|:---------|
| TitleBaseline            | 0.1      | **0.02** | **0.09** | **0.09**    | 0.33     | 0.07     |
| MostCommonAnswerBaseline | 0.03     | 0.0      | 0.03     | 0.03        | 0.0      | 0.01     |
| CosineSimilarityBaseline | **0.11** | 0.02     | 0.09     | 0.09        | **1.36** | **0.12** |
| EditDistanceBaseline     | 0.08     | 0.02     | 0.07     | 0.07        | 1.02     | 0.07     |

