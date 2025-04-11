import pandas as pd
import dspy


def load_data(file_path):
    data = pd.read_csv(file_path)
    print(data.columns)
    data = data.drop(
        columns=["County", "Health level", "LLAMA", "GEMINI", "GPT4.0", "DDX SNOMED"],
        axis=1,
    )
    dataset = data.to_dict(orient="records")
    return dataset


class PredictionModel(dspy.Signature):
    master_index: str = dspy.InputField(desc="Master Index")
    prompt: str = dspy.InputField(desc="Input Prompt from Nurse")
    nursing_competency: str = dspy.InputField(desc="Nursing Competency")
    clinical_panel: str = dspy.InputField(desc="Clinical Panel")
    clinician: str = dspy.OutputField(desc="Clinician Summary")


summary_generator = dspy.ChainOfThought(PredictionModel)
dataset = load_data("data/train.csv")

train_data = dataset[: int(len(dataset) * 0.25)]
test_data = dataset[int(len(dataset) * 0.25) :]

train_set = [
    dspy.Example(PredictionModel, **data).with_inputs(
        "Master_Index", "Prompt", "Nursing Competency", "Clinical Panel"
    )
    for data in train_data
]
test_set = [
    dspy.Example(PredictionModel, **data).with_inputs(
        "Master_Index", "Prompt", "Nursing Competency", "Clinical Panel"
    )
    for data in test_data
]
