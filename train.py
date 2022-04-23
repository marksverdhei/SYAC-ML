import os
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

from utils import (
    DocumentPreprocessor,
    SampleGenerationCallback,
    read_config,
    get_model_parameters,
)


class SYACDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, preprocessor) -> None:
        df = pd.read_csv(dataset_path)
        
        self.tokenizer = tokenizer
        self.length = len(df)
        self.index = list(df.index)

        self.texts = [preprocessor(row.title, row.body) for _, row in df.iterrows()]
        self.labels = [row.target for _, row in df.iterrows()]

    def __getitem__(self, index):
        inputs = self.tokenizer(self.texts[index], return_tensors="pt", truncation=True)
        labels = self.tokenizer(self.labels[index], return_tensors="pt", truncation=True).input_ids
        data = inputs
        data["label_ids"] = labels
        return data

    def __len__(self):
        return self.length


def get_token_tensor_list(data, key):
    return [x[key].squeeze(0) for x in data]


def collate_data(data):
    """
    Collate function used for the trainer object
    """
    batch_dict = {
        key: pad_sequence(get_token_tensor_list(data, key), batch_first=True)
        for key in ("input_ids", "attention_mask", "label_ids")
    }

    batch_dict["labels"] = batch_dict["label_ids"]
    del batch_dict["label_ids"]
    return batch_dict


def train_model(
    train_set_path, 
    val_set_path, 
    model_conf,
    preprocessor_conf,
    train_conf,
) -> None:
    """
    Trains and checkpoints the model with intermediate evaluations
    Parameters are read from utils.toml
    """
    model_name = model_conf["name"]
    tokenizer_path = model_conf["tokenizer_path"]
    model_path = model_conf["model_path"]

    use_early_stopping = train_conf.get("early_stopping") is not None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    preprocessor = DocumentPreprocessor(preprocessor_conf)

    train_data = SYACDataset(train_set_path, tokenizer, preprocessor)
    val_data = SYACDataset(val_set_path, tokenizer, preprocessor)


    print("=" * 20, "\n")
    print("Training", model_name)
    print("Model parameters:", get_model_parameters(model) // 1e6, "million")
    print("\n" + "=" * 20 + "\n")

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"./checkpoints/{model_name}",
        logging_dir=f"./train_logs/{model_name}",
        load_best_model_at_end=use_early_stopping,
        **train_conf["training_args"],
    )

    callbacks = [
        SampleGenerationCallback(
            val_set_path,
            tokenizer,
            preprocessor
        ),
    ]

    if use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(**train_conf["early_stopping"]),
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collate_data,
        callbacks=callbacks,
    )

    out_dir = training_args.output_dir
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        resume_from_checkpoint = True
    else:
        resume_from_checkpoint = False

    train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if training_args.save_strategy.value != "epoch":
        trainer.save_model()
    print(train_output)

    if training_args.evaluation_strategy.value != "epoch":
        eval_post_train = trainer.evaluate()
        print(eval_post_train)
        

def main():
    conf = read_config()

    train_model(
        train_set_path=conf["dataset"]["train_path"],
        val_set_path=conf["dataset"]["val_path"],
        train_conf=conf["train"],
        model_conf=conf["model"],
        preprocessor_conf=conf["preprocessor"],
    )


if __name__ == "__main__":
    main()
