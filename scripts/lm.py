import dspy

lm = dspy.LM(
    "openai/EXTERNAL:google/gemini-2.0-flash",
    api_key="PROVIDER_API_KEY",
    api_base="https://llm-gateway.internal.latest.acvauctions.com/openai/v1",
)
dspy.configure(lm=lm)
lm("Say this is a test!", temperature=0.7)
