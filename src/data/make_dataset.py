from datasets import Dataset
import glob
import pandas as pd
from datasets import concatenate_datasets

DIR_HUM = "data/external/human_written/"
DIR_SYNTH = "data/external/synthetic/"


def make_train_dataset():
    final_ds = concatenate_datasets((human_dataset(), synth_dataset()))
    final_ds.save_to_disk("data/interim/joint_dataset")


def human_dataset():
    brown_dataset = Dataset.from_csv(f"{DIR_HUM}brown.csv")
    brown_dataset = brown_dataset.select_columns(["tokenized_text"])
    brown_dataset = brown_dataset.rename_column("tokenized_text", "text")

    ellipse_dataset = Dataset.from_csv(f"{DIR_HUM}ELLIPSE_Final_github.csv")
    ellipse_dataset = ellipse_dataset.select_columns(["full_text"])
    ellipse_dataset = ellipse_dataset.rename_column("full_text", "text")

    feedback_prize_dataset = Dataset.from_csv(f"{DIR_HUM}feedback_prize_eff_arg.csv")
    feedback_prize_dataset = feedback_prize_dataset.select_columns(
        ["discourse_text"]
    ).rename_column("discourse_text", "text")

    wiki_dataset = Dataset.from_parquet(f"{DIR_HUM}/wikipedia_subset.parquet")
    wiki_dataset = (
        wiki_dataset.select_columns(["text"]).shuffle(seed=42).select(range(40000))
    )

    ghb_list = glob.glob(f"{DIR_HUM}/human_ghostbuster/*.txt")
    ghb_contents = []
    for file_path in ghb_list:
        with open(file_path, "r", encoding="utf-8") as file:
            ghb_contents.append(file.read())

    ghostbuster_dataset = Dataset.from_pandas(
        pd.DataFrame(ghb_contents, columns=["text"])
    )

    human_dataset = concatenate_datasets(
        (
            brown_dataset,
            ellipse_dataset,
            feedback_prize_dataset,
            wiki_dataset,
            ghostbuster_dataset,
        )
    )
    print(f"human datset {len(human_dataset)}")
    generated_flag = [0] * len(human_dataset)
    human_dataset = human_dataset.add_column("generated", generated_flag)
    return human_dataset


def synth_dataset():
    daist_dataset = Dataset.from_csv(f"{DIR_SYNTH}/daigst_v2.csv")
    daist_dataset = daist_dataset.select_columns(["text"])

    t5_essay_dataset = Dataset.from_csv(f"{DIR_SYNTH}/t5_essays_processed.csv")
    t5_essay_dataset = t5_essay_dataset.rename_column(
        "essay_text", "text"
    ).select_columns(["text"])

    mlm_essay_dataset = Dataset.from_csv(f"{DIR_SYNTH}/mlm_essays_processed.csv")
    mlm_essay_dataset = mlm_essay_dataset.rename_column(
        "essay_text", "text"
    ).select_columns(["text"])

    ghb_synth_list = glob.glob(f"{DIR_SYNTH}synth_grohstbuster/*/*.txt")
    ghb_synth_contents = []
    for file_path in ghb_synth_list:
        with open(file_path, "r", encoding="utf-8") as file:
            ghb_synth_contents.append(file.read())

    ghostbuster_synth_dataset = Dataset.from_pandas(
        pd.DataFrame(ghb_synth_contents, columns=["text"])
    )

    synth_dataset = concatenate_datasets(
        (daist_dataset, mlm_essay_dataset, t5_essay_dataset, ghostbuster_synth_dataset)
    )
    generated_flag = [1] * len(synth_dataset)
    synth_dataset = synth_dataset.add_column("generated", generated_flag)
    print(f"synthetic datset {len(synth_dataset)}")
    return synth_dataset


if __name__ == "__main__":
    print(make_train_dataset())
