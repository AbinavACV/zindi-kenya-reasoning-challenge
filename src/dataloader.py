import pandas as pd
import dspy


class PredictionModel(dspy.Signature):
    Prompt: str = dspy.InputField(desc="Input Prompt from Nurse")
    Clinician: str = dspy.OutputField(desc="Clinician Summary")


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        data = pd.read_csv(self.file_path)
        print(data.columns)
        data = data.drop(
            columns=[
                "County",
                "Health level",
                "LLAMA",
                "GEMINI",
                "GPT4.0",
                "DDX SNOMED",
                "Master_Index",
                "Years of Experience",
                "Nursing Competency",
                "Clinical Panel",
            ],
            axis=1,
        )
        dataset = data.to_dict(orient="records")
        return dataset

    def split_data(self, dataset, split_ratio=0.2):
        train_data = dataset[: int(len(dataset) * split_ratio)]
        test_data = dataset[int(len(dataset) * split_ratio) :]
        return train_data, test_data

    def create_examples(self, data):
        return [
            dspy.Example(PredictionModel, **data).with_inputs("Prompt") for data in data
        ]

    def get_data(self):
        dataset = self.load_data()
        train_data, test_data = self.split_data(dataset)
        train_set = self.create_examples(train_data)
        test_set = self.create_examples(test_data)
        return train_set, test_set
