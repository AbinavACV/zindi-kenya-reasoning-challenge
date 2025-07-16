import dspy
from src.metric import dspy_rouge
from src.dataloader import DataLoader
from src.dataloader import PredictionModel

lm = dspy.LM(
    "openai/EXTERNAL:google/gemini-2.0-flash-lite-001",
    api_base="https://llm-gateway.internal.latest.acvauctions.com/openai/v1",
    api_key="xxx",
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

copro_optimizer = dspy.COPRO(metric=dspy_rouge, init_temperature=0.1)

mipro_optimized_summary = mipro_optimizer.compile(
    summary_generator,
    trainset=train_set,
    max_bootstrapped_demos=4,
    requires_permission_to_run=False,
    minibatch=False,
)

# copro_optimized_summary = copro_optimizer.compile(
#     summary_generator,
#     trainset=train_set,
#     eval_kwargs= {}
# )


evaluate(summary_generator, devset=test_set)
mipro_optimized_summary.save("prompts/challenge_mipro_prompt_v1.json")
# copro_optimized_summary.save("prompts/challenge_copro_prompt_v1.json")
