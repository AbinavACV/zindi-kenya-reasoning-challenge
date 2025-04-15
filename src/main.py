import dspy
from src.metric import dspy_rouge
from src.dataloader import DataLoader
from src.dataloader import PredictionModel

lm = dspy.LM(
    "ollama_chat/phi4-mini:latest", api_base="http://localhost:11434", api_key=""
)
dspy.configure(lm=lm)

train_set, test_set = DataLoader("data/train.csv").get_data()
summary_generator = dspy.ChainOfThought(PredictionModel)

# print(test_set[0])
evaluate = dspy.Evaluate(
    devset=test_set,
    metric=dspy_rouge,
    num_threads=1,
    display_progress=True,
    display_table=True,
)

mipro_optimizer = dspy.MIPROv2(
    metric=dspy_rouge,
    auto="medium",
)

optimized_summary = mipro_optimizer.compile(
    summary_generator,
    trainset=train_set,
    max_bootstrapped_demos=1,
    requires_permission_to_run=False,
    minibatch=False,
)
evaluate(summary_generator, devset=test_set)
optimized_summary.save("prompts/challenge_prompt_v1.json")
