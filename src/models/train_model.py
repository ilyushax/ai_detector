import click
from datasets import Dataset
from transformers import DebertaForSequenceClassification, DebertaTokenizer
from ..utils.trainer import tokenize_dataset, get_trainer, get_lora_trainer


@click.command()
@click.option('--input', default="data/interim/joint_dataset")
@click.option('--output', default="models/base")
@click.option('--lora', cls=bool, default=False)
def train(input, output, lora):
    model = DebertaForSequenceClassification.from_pretrained(
        "microsoft/deberta-base", num_labels=2
    )
    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")

    dataset = Dataset.load_from_disk(input)
    dataset = tokenize_dataset(dataset, tokenizer)

    if lora:
        trainer = get_lora_trainer(model, dataset, output)
    else:
        trainer = get_trainer(model, dataset, output)

    trainer.train()
    trainer.save()


if __name__=='__main__':
    train()
