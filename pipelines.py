import os
from abc import ABC, abstractmethod

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
from utils import DocumentPreprocessor, get_best_checkpoint

MAX_GEN_LEN = 128

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
        model_name = config["model"]["name"]
        tokenizer_path = config["model"]["tokenizer_path"]
        preprocessor_conf = config.get("preprocessor", {})

        output_dir = f"./checkpoints/{model_name}/"
        if os.path.isdir(output_dir):
            best_checkpoint = get_best_checkpoint(output_dir)
        else:
            best_checkpoint = config["model"]["model_path"]

        if not config["model"].get("extractive_qa"):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(best_checkpoint)

            return AbstractiveTAPipeline(
                model_name=model_name,
                preprocessor=DocumentPreprocessor(preprocessor_conf),
                tokenizer=tokenizer,
                model=model,
            )
        else:
            return ExtractiveQAPipeline(
                model_name=model_name,
                model_path=best_checkpoint,
                tokenizer_path=tokenizer_path,
            )


class AbstractiveTAPipeline(TitleAnsweringPipeline):
    """
    Loads a pretrained Seq2Seq model
    """

    def __init__(
        self, 
        model_name, 
        preprocessor, 
        tokenizer, 
        model, 
        max_input_length=None, 
        max_generation_length=None,
    ) -> None:
        self.name = model_name
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.model = model
        self.tokenizer.model_max_length = max_input_length
        self.model.config.max_length = max_generation_length or MAX_GEN_LEN

    @torch.no_grad()
    def __call__(self, title, body) -> str:
        input_str = self.preprocessor(title, body)
        inputs = self.tokenizer(input_str, return_tensors="pt", truncation=True).to(
            self.model.device
        )
        (generated_tokens,) = self.model.generate(
            **inputs, num_beams=1, do_sample=False,
        ).to("cpu")
        output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return output


class ExtractiveQAPipeline(TitleAnsweringPipeline):
    """
    Model for title answering based on a QA model huggingface pipeline
    """

    def __init__(self, model_name, model_path, tokenizer_path):
        self.name = model_name
        self.pipeline = pipeline(
            "question-answering", model=model_path, tokenizer=tokenizer_path,
        )
        self.tokenizer = self.pipeline.tokenizer

    def __call__(self, title, body) -> str:
        result = self.pipeline({"question": title, "context": body})
        return result["answer"]
