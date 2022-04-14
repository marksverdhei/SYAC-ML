from utils import (
    DocumentPreprocessor,
    read_config,
    read_tsv,
    get_model_parameters,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    set_seed,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

class SYACDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, preprocessor) -> None:
        df = read_tsv(dataset_path)
        
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
    training_args,
    preprocessor_conf,
    model_path,
    tokenizer_path=None,
    **kwargs,
) -> None:
    """
    Trains and checkpoints the model with intermediate evaluations
    Parameters are read from utils.toml
    """
    if tokenizer_path is None:
        tokenizer_path = model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    preprocessor = DocumentPreprocessor(preprocessor_conf)

    train_data = SYACDataset(train_set_path, tokenizer, preprocessor)
    val_data = SYACDataset(val_set_path, tokenizer, preprocessor)


    print("=" * 20, "\n")
    print("Model parameters:", get_model_parameters(model))
    print("\n" + "=" * 20 + "\n")

    training_args = Seq2SeqTrainingArguments(**training_args)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collate_data,
    )

    # eval_pre_train = trainer.evaluate()
    # print(eval_pre_train)
    train_output = trainer.train()
    trainer.save_model()
    print(train_output)

    if training_args.evaluation_strategy.value != "epoch":
        eval_post_train = trainer.evaluate()
        print(eval_post_train)
        

def main():
    # TODO
    set_seed(42)
    conf = read_config(["train", "datapaths", "preprocessor"])
    train_conf = conf["train"]
    datapaths = conf["datapaths"]

    train_model(
        train_set_path=datapaths["train_path"],
        val_set_path=datapaths["val_path"],
        # training_args=train_conf["training_args"],
        # model_path=train_conf["model_path"],
        preprocessor_conf=conf["preprocessor"],
        **train_conf,
    )


if __name__ == "__main__":
    main()
