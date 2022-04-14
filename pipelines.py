import re
from abc import ABC, abstractmethod

from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration
import torch
"""
The main motivation of this module is to provide abstraction of the models
to make the classificiation (and prehaps even the training scripts) code model agnostic.
For this, it is important that the all the objects implement the same interface
"""


class TitleAnsweringPipeline:
    @staticmethod
    def from_config(config):
        # TODO: implement
        raise NotImplementedError("Implement this")


class TitleAnsweringPipelineBase(ABC):
    """
    This class outlines a functional interface for a 
    title answerer it should implement the call method which
    takes in the title and body of a document and outputs the answer
    """

    @abstractmethod
    def __call__(self, title, body) -> str:
        pass


class TAExtractiveQAPipeline(TitleAnsweringPipelineBase):
    """
    Model for title answering based on a QA model huggingface pipeline
    """

    def __init__(self, hf_pipeline):
        self._internal_pipeline = hf_pipeline

    def __call__(self, title, body) -> str:
        return self.model_output(title, body)["answer"]

    def model_output(self, title, body):
        return self._internal_pipeline({"question": title, "context": body})


class TitleAnsweringUnifiedQAPipeline(TitleAnsweringPipelineBase):
    """
    Model for title answering based on a QA model huggingface pipeline
    """

    def __init__(self, model_path, tokenizer_path):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    @torch.no_grad()
    def __call__(self, title, body) -> str:
        input_str = self.preprocess(title, body)
        tokens = self.tokenizer.encode(input_str, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        model_output = self.model.generate(tokens).to("cpu")
        resp, = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
        return resp

    def preprocess(self, title, body) -> str:
        combined_str = title.lower() + r" \n " + body.lower()
        input_str = re.sub("'(.*)'", r"\1", combined_str)
        return input_str
