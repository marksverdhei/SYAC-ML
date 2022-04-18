import re
from abc import ABC, abstractmethod

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration
import torch
from utils import DocumentPreprocessor, get_best_checkpoint
"""
The main motivation of this module is to provide abstraction of the models
to make the classificiation (and prehaps even the training scripts) code model agnostic.
For this, it is important that the all the objects implement the same interface
"""


class TitleAnsweringPipeline(ABC):
    """
    This class outlines a functional interface for a 
    title answerer it should implement the call method which
    takes in the title and body of a document and outputs the answer
    """

    @abstractmethod
    def __call__(self, title: str, body: str) -> str:
        pass


    @staticmethod
    def from_config(config):
        output_dir = config["train"]["training_args"]["output_dir"]
        tokenizer_path = config["train"]["tokenizer_path"]
        best_checkpoint = get_best_checkpoint(output_dir)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(best_checkpoint)

        return AbstractiveTAPipeline(
            preprocessor=DocumentPreprocessor(config["preprocessor"]),
            tokenizer=tokenizer,
            model=model,
        )


class AbstractiveTAPipeline(TitleAnsweringPipeline):    
    def __init__(self, preprocessor, tokenizer, model, max_length=None) -> None:
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.model = model
        self.truncation = max_length == True
        self.max_length = max_length if self.truncation == False else None

    @torch.no_grad()
    def __call__(self, title, body) -> str:
        input_str = self.preprocessor(title, body)
        inputs = self.tokenizer(input_str, return_tensors="pt").to(self.model.device)
        generated_tokens, = self.model.generate(**inputs).to("cpu")
        output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return output


class TAExtractiveQAPipeline(TitleAnsweringPipeline):
    """
    Model for title answering based on a QA model huggingface pipeline
    """

    def __init__(self, hf_pipeline):
        self._internal_pipeline = hf_pipeline

    def __call__(self, title, body) -> str:
        return self.model_output(title, body)["answer"]

    def model_output(self, title, body):
        return self._internal_pipeline({"question": title, "context": body})
