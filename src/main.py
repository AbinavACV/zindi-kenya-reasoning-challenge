import dspy
from src.metric import rouge_metric
from src.dataloader import train_set, test_set
from src.dataloader import PredictionModel

lm = dspy.LM(
    "ollama_chat/phi4-mini:latest", api_base="http://localhost:11434", api_key=""
)
dspy.configure(lm=lm)

summary_generator = dspy.ChainOfThought(PredictionModel)

evaluate_correctness = dspy.Evaluate(
    devset=test_set,
    metric=rouge_metric,
    num_threads=4,
    display_progress=True,
    display_table=True,
)

mipro_optimizer = dspy.MIPROv2(
    metric=rouge_metric,
    auto="medium",
)
optimized_summary = mipro_optimizer.compile(
    summary_generator,
    trainset=train_set,
    max_bootstrapped_demos=1,
    requires_permission_to_run=False,
    minibatch=False,
)
evaluate_correctness(summary_generator, devset=test_set)
optimized_summary.save("challenge_prompt_v1.json")
