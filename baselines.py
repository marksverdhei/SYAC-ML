from pipelines import TitleAnsweringPipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics.distance import edit_distance
from functools import partial
import nltk


class TitleBaseline(TitleAnsweringPipeline):
    """
    This baseline returns the title.
    The baseline is totally useless for title answering but
    gives a lower bound for the expected seq2seq score
    """
    name = "TitleBaseline"
    def __call__(self, title, body) -> str:
        return title


class MostCommonAnswerBaseline(TitleAnsweringPipeline):
    """
    This baseline returns the most common title answer: 'no'
    """
    name = "MostCommonAnswerBaseline"
    def __call__(self, title, body) -> str:
        return "no"


class CosineSimilarityBaseline(TitleAnsweringPipeline):
    name = "CosineSimilarityBaseline"
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


class EditDistanceBaseline(TitleAnsweringPipeline):
    name = "EditDistanceBaseline"
    def __call__(self, title, body) -> str:
        body_sentences = nltk.tokenize.sent_tokenize(body)
        return min(body_sentences, key=partial(edit_distance, title))


BASELINE_CLASSES = [
    TitleBaseline,
    MostCommonAnswerBaseline,
    CosineSimilarityBaseline,
    EditDistanceBaseline,
]

BASELINES = {
    baseline.__name__: baseline() for baseline in BASELINE_CLASSES 
}