import openai

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

# List models API
models = openai.Model.list()
print("Models:", models)

model = models["data"][0]["id"]

# Completion API
stream = False
completion = openai.Completion.create(
    model=model,
    prompt="A robot may not injure a human being",
    echo=False,
    n=2,
    stream=stream,
    logprobs=3)

# Save results to a file
with open("results.txt", "w") as f:
    f.write("Models:\n")
    f.write(str(models) + "\n")
    
    f.write("\nCompletion results:\n")
    if stream:
        for c in completion:
            f.write(str(c) + "\n")
    else:
        f.write(str(completion))

print("Completion results:")
if stream:
    for c in completion:
        print(c)
else:
    print(completion)
