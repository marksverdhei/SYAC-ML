from pipelines import TitleAnsweringPipelineBase
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics.distance import edit_distance
from functools import partial
import nltk


class TitleBaseline(TitleAnsweringPipelineBase):
    """
    This baseline returns the title.
    The baseline is totally useless for title answering but
    gives a lower bound for the expected seq2seq score
    """
    def __call__(self, title, body) -> str:
        return title


class MostCommonAnswerBaseline(TitleAnsweringPipelineBase):
    """
    This baseline returns the most common title answer: 'no'
    """
    def __call__(self, title, body) -> str:
        return "no"


class CosineSimilarityBaseline(TitleAnsweringPipelineBase):
    vectorizer_args = {
        "binary": True,
        "stop_words": "english",
    }

    def __call__(self, title, body) -> str:
        vectorizer = CountVectorizer(**self.vectorizer_args)
        vectorizer.fit([body])

        body_sentences = nltk.tokenize.sent_tokenize(body)
        title_vector = vectorizer.transform([title])

        def score_sentence(s):
            sentence_vector = vectorizer.transform([s])
            return cosine_similarity(title_vector, sentence_vector)

        return max(body_sentences, key=score_sentence)


class EditDistanceBaseline(TitleAnsweringPipelineBase):
    def __call__(self, title, body) -> str:
        body_sentences = nltk.tokenize.sent_tokenize(body)
        return min(body_sentences, key=partial(edit_distance, title))


# class ExtractiveRegexBaseline(TitleAnsweringPipelineBase):
#     """
#     The model searches for common patterns
    
#     """
    
#     top_list_pattern = r"((?<=\n)|[\(\[])[0-9]+[\.\]\)\:\; \t].+$"
#     # title_proximity_pattern = 

#     def __call__(self, title, body) -> str:
#         return ""


BASELINE_CLASSES = [
    TitleBaseline,
    MostCommonAnswerBaseline,
    CosineSimilarityBaseline,
    EditDistanceBaseline,
]

BASELINES = {
    baseline.__name__: baseline() for baseline in BASELINE_CLASSES 
}


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("../data/train.csv")
    example = df.sample(1)
    title = example.title.item()
    body = example.body.item()
    target = example.target.item()

    print("="*20+"title"+"="*20)
    print(title)
    print("="*20+"target"+"="*20)
    print(target)
    print("\n")

    for name, baseline in BASELINES.items():
        print("="*20+name+"="*20)
        print(baseline(title, body))