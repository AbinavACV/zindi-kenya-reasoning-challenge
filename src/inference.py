import dspy
import pandas as pd
from src.dataloader import PredictionModel


def load_csv(file_path: str):
    data = pd.read_csv(file_path)
    return list(zip(data["Master_Index"].to_list(), data["Prompt"].tolist()))


# def get_loaded_prompt(saved_prompt_path: str):
#     generator = dspy.ChainOfThought(PredictionModel)
#     loaded_prompt = generator.load(saved_prompt_path)
#     return loaded_prompt


def perform_inference(loaded_prompt, language_model: str, input_string: str):
    lm = dspy.LM(language_model, api_base="http://localhost:11434", api_key="")
    dspy.configure(lm=lm)
    output = loaded_prompt(Prompt=input_string)
    return output


if __name__ == "__main__":
    language_model = "ollama_chat/smollm:360m"
    saved_prompt_path = "/Users/abinav/Desktop/repos/competitions/zindi-kenya-reasoning-challenge/prompts/challenge_prompt_smollm_v1.json"

    class PredictionModel(dspy.Signature):
        Prompt: str = dspy.InputField(desc="Input Prompt from Nurse")
        Clinician: str = dspy.OutputField(desc="Clinician Summary")

    generator = dspy.ChainOfThought(PredictionModel)
    generator.load(saved_prompt_path)

    # loaded_prompt = get_loaded_prompt(saved_prompt_path)
    data = load_csv("data/test.csv")
    output_dict = {}
    final_list = []
    for idx, input_string in data:
        output = perform_inference(generator, language_model, input_string)
        output_dict["Master_Index"] = idx
        output_dict["Clinician"] = output.Clinician
        final_list.append(output_dict.copy())
    # print(final_list)
    pd.DataFrame(final_list).to_csv("data/submission.csv", index=False)
